import base64
import os
import queue
import re
import threading
import time
import uuid
from typing import TYPE_CHECKING, Callable, Generator, Iterator, Optional

from colorama import Fore

from common.DevelopmentConfig import BYTE_PREFIX_TAIL, prepend_user_query_override
from assistant.DialogManager import DialogManager
from assistant.prompt.MessageHistory import MessageHistory
from assistant.AssistantRequest import RequestContext
from common.RequestKilledException import RequestKilledException
from common.logging import info, trace, note, error, fine
from assistant.PromptInjectionDetector import is_prompt_injection
from assistant.Telemetry import Telemetry
from assistant.tools.MessageFormatter import MessageFormatter
from common.FloxyDatabase import FloxyDatabase
from assistant.AssistantEvents import (AssistantEvent, AudioChunk, Done, Notice, RoundEnd, Token,
                                       observers)
from assistant.commands.OrphanedCommands import orphan_registry
from assistant import FirstRun
from assistant.commands import ToolkitPreloader
from assistant.ChannelProfile import profile_for
from assistant.llm.AbstractLLM import api_error_message

# ServerPlayback, VoiceTurnGate and the TTS synthesizers are only ever constructed by the
# launcher's factories, so this module never imports an interface package.
if TYPE_CHECKING:
    from interfaces.voice.server_audio.ServerPlayback import ServerPlayback
    from interfaces.voice.tts.VoiceSynthesizer import VoiceSynthesizer
    from interfaces.voice.VoiceTurnGate import VoiceTurnGate
    from interfaces.voice.ios.TextStreamRegistry import TextStreamRegistry


def _format_command_reply(reply) -> str:
    if not isinstance(reply, list):
        return str(reply)
    parts = []
    for i, block in enumerate(reply):
        btype = block.get("type", "unknown")
        if btype == "text":
            parts.append(f"  [{i}] text: {block.get('text', '')}")
        elif btype == "image":
            src = block.get("source", {})
            data = src.get("data", "")
            parts.append(f"  [{i}] image: {src.get('media_type', '?')}, {len(data)} chars base64")
        else:
            parts.append(f"  [{i}] {btype}: {block}")
    return "\n".join(parts)


def _command_result_for_history(reply: str | list) -> str:
    """The text of a follow-up round's reply as it goes into the history: a plain reply
    verbatim, a block list as its text blocks with each image noted — images are not
    persisted (a camera snapshot has no business in every later prompt)."""
    if isinstance(reply, str):
        return reply
    parts = []
    for block in reply:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif block.get("type") == "image":
            parts.append("[image not kept in history]")
    return "\n".join(parts)


def _build_error_round_reply(command_reply, executed_commands: list) -> str | list:
    """Frame a command-error follow-up round for the LLM. Informational only: reports the
    failures, which commands already ran, and that the previous reply was already spoken —
    the model decides how to proceed. Without this framing the follow-up round answered the
    whole query again: a second full spoken reply plus re-execution of commands that had
    already succeeded (the 2026-08-03 double-reply incidents)."""
    if executed_commands:
        succeeded_line = "These commands already executed successfully: " + ", ".join(executed_commands) + "."
    else:
        succeeded_line = "No other commands were executed."
    note_line = ("Note: your previous reply was already spoken to the user. Decide the best "
                 "way to proceed given what the user already heard and what actually executed.")
    if isinstance(command_reply, list):
        # The error text block is already in the list (the processor composes it in front).
        preamble = ("Command execution report — some commands from your reply failed "
                    f"(see below).\n{succeeded_line}\n\n{note_line}")
        return [{"type": "text", "text": preamble}] + command_reply
    return (f"Command execution report — some commands from your reply failed:\n"
            f"{command_reply}\n{succeeded_line}\n\n{note_line}")


def _build_repeated_round_reply(command_reply: str) -> str:
    """Frame a round that repeated the previous round's commands to the same result.
    Informational: the result again, and that it is the same as before."""
    return (f"result: \n{command_reply}\n\nThis round ran the same command(s) as the previous round and the "
            f"result is the same as before. Calling again returns it unchanged.")


def _save_image_to_log(image_block: dict, request_id: str, index: int) -> str | None:
    from common.logging import disable_file_logging
    if disable_file_logging:
        return None
    try:
        source = image_block.get("source", image_block)
        data_b64 = source.get("data", "")
        media_type = source.get("media_type", "image/jpeg")
        ext = "png" if "png" in media_type else "jpg"

        os.makedirs("log/images", exist_ok=True)
        filename = f"{request_id}_{index}.{ext}"
        path = f"log/images/{filename}"

        with open(path, "wb") as f:
            f.write(base64.b64decode(data_b64))

        info(f"Image saved: {path}")
        return path
    except Exception as e:
        trace(f"Failed to save image to log: {e}")
        return None


__all__ = ["RequestProcessor", "RequestData", "RequestKilledException"]


class RequestProcessor:
    """Runs one request end to end and yields the AssistantEvents the interface renders.

    The pipeline is prompt → LLM → commands → history. Single-use: Assistant pools fresh
    instances.
    """
    # Set by _prepare_llm_context for a byte-prefix request: when the query was sent (its
    # header carries the time) and the clock note sent in front of it, both persisted with the
    # exchange so the next request renders the same bytes. None otherwise.
    turn_timestamp: float | None = None
    time_note: str | None = None

    # Used by _iter_until_killed to poll the kill flag while the TTS producer is blocked.
    KILL_POLL_INTERVAL = 0.5

    # What the user is told when the LLM backend fails mid-request without a word from the
    # server (transport drop, timeout) — see _llm_failure_notice. Shown as a Notice in text
    # mode, spoken via TTS in voice mode. One constant so the channels never drift.
    LLM_OUTAGE_NOTICE = "The AI servers are unavailable right now. Please try again in a moment."
    LLM_FAILURE_DETAIL_MAX_CHARS = 300
    MAX_UNTERMINATED_COMMAND_REPORTS = 2
    # A follow-up round that runs exactly the previous round's commands and gets the same
    # text back learns nothing: the first repeat is told so under its result, the second
    # ends the request (ds4 at effort low called load_toolkit("setup") four rounds running
    # over a 102-char "it is above" answer until Ctrl-C, 2026-10-22). An image reply is
    # never a repeat — a camera checked again is a new picture under the same text.
    MAX_REPEATED_ROUNDS = 2
    REPEATED_ROUND_NOTICE = "Stopped: the same command returned the same result three rounds in a row."
    LLM_OUTAGE_HISTORY_MARKER = "\n[the AI backend failed mid-reply]"
    # The result of the one extra round a request gets when it would end without a word to
    # the user although commands ran (see _reply_owed). ds4 often writes the bookkeeping call
    # first and the answer never: a research turn ended on checked_watch, the user saw
    # nothing (13 silent Telegram turns on 2026-10-22).
    SILENT_END_NOTE = ("The commands ran, and nothing has been said to the user this turn. "
                       "Reply to the user now.")
    # The note a round that ended at the LLM's output cap circles back with, as its result
    # (see _cap_hit): the thinking counts against the cap on the 5-generation models, and
    # Opus 5.5 at high effort thought through the whole of it three times running on a
    # home-automation plan — the user got silence (2026-10-23, issue #108). The model
    # cannot lower its own effort, so it is asked in words. Once per request: a second cut
    # ends the request with CAP_HIT_NOTICE to the user.
    CAP_HIT_THINKING_NOTE = ("Your reply hit {cap} while still thinking. No text was produced, so nothing "
                             "reached the user and the attempt was discarded. Try again with less deliberation: "
                             "reason briefly and answer, or split the task over several rounds.")
    CAP_HIT_TEXT_NOTE = ("Your reply hit {cap} after: \"{tail}\". The user saw the text up to there. "
                         "Continue from that point, briefly.")
    CAP_HIT_TAIL_CHARS = 120
    CAP_HIT_NOTICE = "The AI's reply was cut short at its output limit twice. Ask again, or in smaller steps."

    def __init__(self, playback_factory: "Callable[[Callable[[], bool]], ServerPlayback] | None" = None,
                 voice_factory: "Callable[[str | None], VoiceSynthesizer] | None" = None,
                 turn_gate_factory: "Callable[[Callable[[], bool], Callable[[], str]], VoiceTurnGate] | None" = None,
                 text_stream: "TextStreamRegistry | None" = None):
        """playback_factory builds the server-speaker playback for this handler, given a
        callable that reports whether the request was killed. None (no voice interface in
        this process) means audio is only ever streamed back to the client. voice_factory
        builds the TTS synthesizer per request language (interfaces.voice.tts.VoiceFactory);
        None means every reply is text-only. turn_gate_factory builds the reply-timing gate
        (interfaces.voice.VoiceTurnGate) the same way; None means the first sound of a reply
        is never held back. text_stream is the iOS /stream_text overlay channel the reply's
        text is mirrored to (interfaces.voice.ios); None means nothing is published."""
        self.voice_factory = voice_factory
        self.text_stream = text_stream
        # Side channel for events raised off the token loop (Thinking chunks from the command
        # processor, CommandRan from the command worker thread); drained into the event stream
        # at every yield point. Shared with the dialog managers this handler builds.
        self.events: "queue.SimpleQueue[AssistantEvent]" = queue.SimpleQueue()
        # Events the TTS feeder (_text_only) pulls out of the LLM stream, parked here for the
        # audio loop. Never `events`: the stream it feeds from drains that very queue, and
        # putting the event back there made the feeder thread spin at 100% — pull, put, pull —
        # for as long as nobody else drained it, i.e. forever once the request was killed.
        self._tts_events: "queue.SimpleQueue[AssistantEvent]" = queue.SimpleQueue()
        self.dialog_manager = self._new_dialog_manager()
        # Every dialog manager this request ran a round on, so the telemetry can name the
        # commands of the whole request, not only the last round's.
        self._rounds: list = []
        self._silent_end_followed_up = False
        self._cap_hit_followed_up = False
        self._cut_round_result: str | None = None  # the second cut round's command result, for the history
        self.first_sound_chunk = True
        self.is_killed = False
        self.kill_reason = ""  # why kill() fired, so the turn gate can log a truthful abort
        self.per_request_prefix = False  # the prepend verdict of the request, for every round's LLM
        self.is_performing_request = False
        self.my_thread_name = "?"
        self.request_data = RequestData(None, None, None, None)
        self.request_context: Optional[RequestContext] = None
        # Reply-timing gate (speech onset / Scribe verdict holds); the microphone routes arm
        # its per-request signals. One per handler; None in a voice-less process.
        self.turn_gate: "VoiceTurnGate | None" = (
            turn_gate_factory(lambda: self.is_killed, lambda: self.kill_reason)
            if turn_gate_factory else None)
        # Server-speaker playback + outpost mute choreography (interfaces/voice/server_audio).
        # One per handler, like the handler itself single-use; None in a voice-less process.
        self.playback: "ServerPlayback | None" = (
            playback_factory(lambda: self.is_killed) if playback_factory else None)

    def _new_dialog_manager(self, test_suite_mode: bool = False) -> DialogManager:
        return DialogManager(self, test_suite_mode, voice_factory=self.voice_factory)

    def _carry_per_request_prefix(self):
        """The prepend verdict onto every tier of the current dialog manager — a follow-up
        round may switch to the vision tier (use_vision_for_image_reply) after select_ai."""
        if not self.per_request_prefix:
            return
        trace("Carrying the per-request prefix verdict into the follow-up round's LLMs")
        for ai in (self.dialog_manager.ai_fast, self.dialog_manager.ai_smart, self.dialog_manager.ai_vision):
            ai.set_per_request_prefix(True)

    @property
    def ctx(self) -> RequestContext:
        return self.request_context

    def kill(self, reason: str = ""):
        if not self.is_killed:
            fine(f"Killing request handler: {self.my_thread_name}" + (f" ({reason})." if reason else "."))
            self.is_killed = True
            self.kill_reason = reason
            self.dialog_manager.kill()
            if self.playback:
                self.playback.stop()

    def _check_killed(self):
        if self.is_killed:
            raise RequestKilledException()

    def emit(self, event: AssistantEvent):
        """Queue an event for the stream (any thread)."""
        self.events.put(event)

    def _drain_events(self) -> Generator[AssistantEvent, None, None]:
        for q in (self._tts_events, self.events):
            while True:
                try:
                    yield q.get_nowait()
                except queue.Empty:
                    break

    def publish_text_event(self, request_id: str | None, event: dict):
        """Mirror a reply event ({"type": "token"|"user", ...}) to the iOS text overlay for
        `request_id`. No-op without a text stream (no voice interface) or request_id."""
        if self.text_stream is not None:
            self.text_stream.publish(request_id, event)

    def _close_text_stream(self, request_id: str | None):
        if self.text_stream is not None:
            self.text_stream.close(request_id)

    @staticmethod
    def _remove_thinking(content: str) -> str:
        """Remove <thinking>...</thinking> tags from content."""
        return re.sub(r"<\s*thinking\s*>(.*?)</thinking>", "", content, flags=re.DOTALL).strip()

    def process(self, prompt_manager) -> Iterator[AssistantEvent]:
        """Prepare the turn and return the event stream (a generator; nothing runs until
        it is iterated). Returns an empty iterator when the handler was killed already."""
        self.is_performing_request = True
        self.my_thread_name = threading.current_thread().name
        if self.turn_gate:
            self.turn_gate.begin_turn(self.request_data.args)

        # Tell the autonomous action manager a user request is in flight so it
        # suppresses periodic/caffeinate/scheduled checks during this turn.
        # Pairing: the matching end-notify lives in generate_audio()'s finally
        # block. Any path here that returns WITHOUT producing the streaming
        # response must call end-notify itself, otherwise the in-flight
        # counter leaks and the autonomous loop is stuck forever.
        self._notify_autonomous_user_started()

        try:
            # set this so that we do not recreate the prompt during or directly after a request,
            # that would zero out the previous dialogue history, and fuck other shit up.
            prompt_manager.prompt_touched_at = time.time()

            reset = 'reset' in self.request_data.args

            if self.ctx.text_only:
                trace("Running text only mode")

            if self.ctx.server_sound:
                fine("Running server sound mode")

            if reset or self.ctx.test:
                self.dialog_manager = self._new_dialog_manager(self.ctx.test)

            r = self.request_data.headers.get('Range', None)
            playback_session = self.request_data.headers.get('X-Playback-Session-Id', None)
            icy_metadata = self.request_data.headers.get('Icy-Metadata', None)
            user_agent = self.request_data.headers.get('User-Agent', '')
            fine(f"user_agent: {user_agent}, range: {r}, playback session: {playback_session}, icy: {icy_metadata}")

            if self.playback:
                self.playback.log_ambient_noise()

            if self.is_killed:
                info("Request was killed before generation started; returning an empty stream.")
                self._notify_autonomous_user_ended()
                return iter(())

            return self.generate_events(prompt_manager)
        except BaseException:
            # generate_events() never got to return its generator, so its finally block
            # won't fire — release the in-flight counter here, and the in-flight flag
            # (Assistant._in_flight reads it: stuck true, it would refuse every guest turn).
            self.is_performing_request = False
            self._notify_autonomous_user_ended()
            raise

    def _handle_first_audio_chunk(self, chunk_time: float):
        info(f"First sound chunk: {chunk_time}s ({self.dialog_manager.voice.impl_name()}).")
        # The cache-hit flag is set before any cached chunk is queued, so it is already accurate here.
        observers().on_first_sound_chunk(chunk_time,
                                         getattr(self.dialog_manager.voice, "last_served_from_cache", False))
        if not self._wait_for_natural_timing(chunk_time):
            raise RequestKilledException()

        self.first_sound_chunk = False
        self._check_killed()

        self._ensure_outposts_muted()

        if not self.ctx.text_only and not (self.playback and self.playback.is_running()):
            note("Starting to speak now.")

    def wait_so_that_response_timing_is_natural(self, chunk_time: float) -> bool:
        """The command worker consults the same gate before running a reply's first command,
        so a command never fires while the user is still talking (see CommandProcessor)."""
        return self._wait_for_natural_timing(chunk_time)

    def _wait_for_natural_timing(self, chunk_time: float) -> bool:
        """Hold the first sound until the voice turn gate says it is natural to speak.
        Runs on the request thread right before the first chunk — no thread hop. Returns
        False when the request was killed while holding."""
        if self.turn_gate is None:
            info("No voice turn gate — speaking immediately.")
            return not self.is_killed
        return self.turn_gate.wait(self.ctx, self.dialog_manager.reply_text,
                                   self.dialog_manager.time_to_first_token, chunk_time)

    def _deliver_audio_chunk(self, chunk: bytes) -> bytes | None:
        # Runs on: request thread, straight from the TTS chunk loop — enqueue() must stay a
        # plain queue.put so no thread hop lands on the time-to-first-sound path.
        if self.ctx.server_sound and self.playback:
            self.playback.enqueue(chunk)
            return None
        fine("Streaming " + str(len(chunk)) + " bytes of audio.")
        return chunk

    def _iter_until_killed(self, gen: Generator) -> Generator:
        """Iterate `gen`, but stop promptly if the request is killed — even when the
        producer is blocked on a slow/hung upstream.

        The audio generator (CachedVoiceSynthesizer.start) blocks on an unbounded
        audio_queue.get(); when a request is superseded mid-flight the TTS impl can
        stop delivering chunks without ever signalling end-of-stream, so that get()
        blocks forever. Iterating it directly means the `for chunk` loop below never
        comes back around to _check_killed(), and the killed request's HTTP response
        stays open until the client's own socket timeout (60s). Pulling on a daemon
        thread and polling is_killed lets us abandon the stuck producer (kill() has
        already torn down the dialog manager and audio playback) and return, so the
        stream ends within KILL_POLL_INTERVAL of the kill."""
        q: "queue.Queue" = queue.Queue(maxsize=8)
        DONE = object()
        exc_holder = [None]

        def produce():
            try:
                for item in gen:
                    while not self.is_killed:
                        try:
                            q.put(("item", item), timeout=self.KILL_POLL_INTERVAL)
                            break
                        except queue.Full:
                            continue
                    if self.is_killed:
                        return
            except Exception as e:
                exc_holder[0] = e
            finally:
                try:
                    q.put((DONE, None), timeout=self.KILL_POLL_INTERVAL)
                except queue.Full:
                    pass

        t = threading.Thread(target=produce, name="audio-pull", daemon=True)
        t.start()

        while not self.is_killed:
            try:
                kind, value = q.get(timeout=self.KILL_POLL_INTERVAL)
            except queue.Empty:
                continue
            if kind is DONE:
                if exc_holder[0]:
                    raise exc_holder[0]
                return
            yield value

    def _llm_failure_notice(self, e: Exception) -> str:
        """The one line every channel shows or speaks when the LLM backend fails: the
        server's own message when it sent one ("The Claude server returned an error: Your
        credit balance is too low…" — the 2026-09-08 billing block would otherwise have read
        as an outage), else the exception's own message (a missing api_key never reaches a
        server, yet says exactly what to fix), and the generic outage line only when the
        exception carries no text at all."""
        provider = self.dialog_manager.ai.provider()
        detail = api_error_message(e)
        if detail is not None:
            return f"The {provider} server returned an error: {detail}"
        detail = " ".join(str(e).split())
        if not detail:
            trace(f"No message in {type(e).__name__} — using the generic outage notice.")
            return self.LLM_OUTAGE_NOTICE
        if len(detail) > self.LLM_FAILURE_DETAIL_MAX_CHARS:
            trace(f"Capping the {len(detail)}-char {type(e).__name__} message in the failure notice.")
            detail = detail[:self.LLM_FAILURE_DETAIL_MAX_CHARS].rstrip() + "…"
        trace(f"No server message in {type(e).__name__} — showing the exception's own message.")
        return f"The {provider} request failed: {detail}"

    def _text_only(self, llm_stream: Iterator) -> Generator[str, None, None]:
        """Feed the TTS synthesizer the text of an LLM stream; the in-stream events ride
        the side channel and surface at the next audio chunk.

        This runs on the TTS collect_text thread, so a backend error escaping here never
        reaches generate_events' text-mode handler — it used to die as an unhandled thread
        exception and the reply just ended silently (the 2026-08-27 Gemini 503). Catch it
        at the same altitude text mode does and speak the same outage notice instead."""
        try:
            for item in llm_stream:
                if isinstance(item, AssistantEvent):
                    self._tts_events.put(item)
                else:
                    yield item
        except RequestKilledException:
            raise
        except Exception as e:
            error(f"LLM stream failed mid-voice-reply ({type(e).__name__}: {e}) — speaking the failure notice.")
            notice = self._llm_failure_notice(e)
            if self.dialog_manager.first_token_got_at == 0:
                self.dialog_manager.first_token_got_at = time.time()
                self.dialog_manager.reply_token_got_at = self.dialog_manager.first_token_got_at
            self.dialog_manager.llm_response += notice
            self.dialog_manager.llm_response_raw += self.LLM_OUTAGE_HISTORY_MARKER
            self.dialog_manager._surface_notice(notice)
            yield notice

    def stream_response(self) -> Generator[AssistantEvent, None, None]:
        """One LLM round as events: Token (text mode) or AudioChunk (client-played audio;
        server-played audio goes straight to the playback and yields nothing), with the
        Thinking / ThinkingDone / CommandRan side events interleaved where they happen."""
        self.first_sound_chunk = True

        text_mode = self.ctx.text_only or self.dialog_manager.voice is None
        if text_mode and not self.ctx.text_only:
            info("No voice synthesizer for this request — streaming the reply as text.")

        if text_mode:
            for chunk in self.dialog_manager.stream_llm_response():
                if isinstance(chunk, AssistantEvent):
                    yield chunk
                elif chunk:
                    yield Token(chunk)
                yield from self._drain_events()
        else:
            audio_stream = self.dialog_manager.voice.start(self._text_only(self.dialog_manager.stream_llm_response()))
            for chunk in self._iter_until_killed(audio_stream):
                if chunk is not None:
                    self._check_killed()
                    if self.first_sound_chunk:
                        chunk_time = round(time.time() - self.dialog_manager.reply_token_got_at, 2)
                        self._handle_first_audio_chunk(chunk_time)
                    result = self._deliver_audio_chunk(chunk)
                    if result is not None:
                        yield AudioChunk(result)
                yield from self._drain_events()

        self._check_killed()

        trace("")
        info(self.dialog_manager.ai.name() + ":")
        note(self.dialog_manager.llm_response.strip())
        trace("")
        info(Fore.LIGHTBLACK_EX + "Raw: ")
        info(Fore.LIGHTBLACK_EX + self._remove_thinking(self.dialog_manager.llm_response_raw))

        self._check_killed()
        yield from self._drain_events()

    def initialize_audio_playback(self):
        if not self.ctx.server_sound or not self.playback:
            return
        self.playback.ensure_started()

    def _inject_orphaned_command_outcomes(self, prompt_manager):
        """Tell the LLM what became of background commands its killed predecessor left
        running (see assistant/commands/OrphanedCommands.py) — without this the model
        has a hole in its history where a command ran and demonstrably fills it with
        invented explanations."""
        if self.ctx.test:
            return
        note_text = orphan_registry().drain_note()
        if note_text is None:
            return
        info(f"Injecting orphaned command outcome into history ({len(note_text)} chars).")
        prompt_manager.message_history.add_transient_entry(
            role="system", content=note_text, timestamp=time.time(), source="orphaned_command")

    def _preload_toolkits(self, prompt_manager):
        """The channel's toolkits (ChannelProfile.preload_toolkits) and the FTUE's
        (FirstRun.preload_toolkits) into the window before the messages are assembled. Not
        in test-suite mode: every toolkit is appended to the fixture history there
        (PromptManager._all_toolkits_loaded)."""
        if self.ctx.test:
            trace("Toolkit preload skipped: test-suite request, every toolkit is in the fixture history")
            return
        ToolkitPreloader.preload(prompt_manager, profile_for(self.ctx.source), FirstRun.preload_toolkits(),
                                 is_killed=lambda: self.is_killed)

    def _prepare_llm_context(self, prompt_manager) -> str | None:
        """Builds the round's messages and returns the key the response cache and the DB row
        use for this query (the formatted query without the byte-prefix header's time), or
        None when the request is denied."""
        self.dialog_manager.select_ai(self.ctx)
        self.dialog_manager.select_voice(self.ctx)

        self.dialog_manager.request_id = self.request_data.args.get('request_id') if self.request_data else None
        if self.dialog_manager.request_id:
            trace(f"Text stream correlated with request_id {self.dialog_manager.request_id}")

        user_input = self.ctx.user_input

        if is_prompt_injection(user_input):
            return None

        # The <thinking>-tag instruction (PromptManager.get_messages) is how a model without
        # thinking of its own reasons before it answers, with the tag stripped before TTS. A
        # model that handles thinking internally must never see it: Claude refuses the whole
        # request as reasoning extraction (AbstractLLM.handles_thinking_internally).
        ai = self.dialog_manager.ai
        # A byte-prefix tail (DevelopmentConfig.BYTE_PREFIX_TAIL): the trailing turn is sent
        # exactly as the history will render it afterwards — clock note as its own row, query
        # header timestamped — and carries nothing the next turn drops, so the previous
        # request's KV-cache restore point stays inside this one's prefix.
        byte_prefix_tail = BYTE_PREFIX_TAIL and ai.prefers_byte_prefix_tail()
        self.dialog_manager.byte_prefix_tail = byte_prefix_tail
        use_prompt_hack = not ai.handles_thinking_internally() and not byte_prefix_tail
        if byte_prefix_tail:
            trace(f"{ai.name()} gets a byte-prefix tail — no <thinking> tag instruction, no trailing directives")
        elif use_prompt_hack:
            trace(f"{ai.name()} gets the <thinking> tag instruction")
        else:
            trace(f"{ai.name()} thinks on its own — no <thinking> tag instruction")
        # Native thinking stays off: these models think adaptively without the parameter.
        use_native_thinking = False

        self._inject_orphaned_command_outcomes(prompt_manager)
        self._preload_toolkits(prompt_manager)
        # A speculative partial-transcript request is usually killed by its successor within
        # milliseconds; nothing after this point is worth doing for one.
        self._check_killed()
        if byte_prefix_tail:
            self.turn_timestamp = time.time()
        if byte_prefix_tail or self.ctx.note:
            self.time_note = self._turn_note(prompt_manager, byte_prefix_tail)
        self.dialog_manager.messages = prompt_manager.get_messages(extended_thinking=use_prompt_hack,
                                                                   time_note=self.time_note,
                                                                   window=profile_for(self.ctx.source).window).copy()
        self.dialog_manager.thinking = use_native_thinking

        source = self.ctx.source
        location = self.ctx.location
        user_input_sent_to_ai = MessageFormatter.format_message(self.ctx.display_name, self.turn_timestamp,
                                                                source, location, user_input)
        # The response cache and the DB row are keyed without the byte-prefix header's HH:MM —
        # the same utterance a minute later must still hit.
        cache_key = (MessageFormatter.format_message(self.ctx.display_name, None, source, location, user_input)
                     if self.turn_timestamp is not None else user_input_sent_to_ai)
        fine("Will send to LLM: " + user_input_sent_to_ai)

        override = prepend_user_query_override()
        should_prepend = override if override is not None else self.dialog_manager.ai.prefers_prepended_user_query()
        # Prepending puts per-request content at byte 0, so no later request reads what this
        # one caches — only its own command follow-up rounds do. The LLM picks its cache tier
        # by that (ClaudeLLM: 5m on every block instead of 1h; a no-op for the others).
        self.per_request_prefix = should_prepend
        self.dialog_manager.ai.set_per_request_prefix(should_prepend)
        if should_prepend:
            trace("Prepending user query - the prompt prefix is this request's own")
            # Prepend user query preview at the top of the system prompt - this improves accuracy significantly (see https://x.com/aakashgupta/status/2023998053737541636?s=20)
            messages = self.dialog_manager.messages
            if messages and messages[0].get("role") == "system":
                heads_up = f"Just a heads up, the user query you will be handling is:\n\n{user_input_sent_to_ai}"
                messages[0] = {"role": "system", "content": heads_up + "\n\n---\n\n" + messages[0]["content"]}

        if self.ctx.images:
            request_id = str(uuid.uuid4())[:8]
            content_blocks = []
            for i, img in enumerate(self.ctx.images):
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img.get("media_type", "image/jpeg"),
                        "data": img["data"],
                    }
                })
                _save_image_to_log(content_blocks[-1], request_id, i)
            content_blocks.append({"type": "text", "text": user_input_sent_to_ai})
            self.dialog_manager.reply_to_llm = content_blocks
        else:
            self.dialog_manager.reply_to_llm = user_input_sent_to_ai
            # Key the response cache on the clean utterance — the same string stored as the
            # DB query in _save_to_database — so lookups can actually match. Image turns are
            # left uncached (cache_query stays None).
            self.dialog_manager.cache_query = cache_key
        return cache_key

    # Runs on: request thread. Blocks until command worker thread finishes processing queued command.
    def _wait_for_initial_command(self):
        cp = self.dialog_manager.command_processor
        self._wait_until(
            lambda: cp.command_being_processed or not cp.command_queue.empty(),
            start_message="Waiting for command to complete...",
            done_message="Command completed.",
        )

    # Runs on: request thread. Blocks until bg command worker thread finishes.
    def _wait_for_background_command(self, old_dm):
        self._wait_until(
            lambda: old_dm.command_processor.command_running_in_background,
            start_message="Waiting until command running in background is done...",
            done_message="Commands done! Let's continue.",
        )

    # Runs on: request thread. Blocks until bg command worker thread sets command_reply.
    def _wait_for_command_reply(self, old_dm):
        self._wait_until(
            lambda: old_dm.command_processor.command_reply is None,
            start_message="Will have to wait until we have command reply...",
            done_message="Commands reply here! Let's continue.",
        )

    def _wait_until(self, should_wait, start_message: str, done_message: str):
        if not should_wait():
            return
        trace(start_message)
        while should_wait():
            self._check_killed()
            time.sleep(0.01)
        trace(done_message)

    def _turn_note(self, prompt_manager, byte_prefix_tail: bool) -> str | None:
        """The note row sent in front of the query: the clock on a byte-prefix request, and
        under it the facts the chat layer states about the message (a guest, a group chat).
        They stand in the note, not in the person's own text, so the LLM reads them as the
        system's words and not as something the person wrote."""
        clock = prompt_manager.current_time_note(self.turn_timestamp) if byte_prefix_tail else None
        facts = self.ctx.note
        if not facts:
            return clock
        trace(f"Turn note carries the chat layer's facts about the message ({len(facts)} chars)")
        return f"{clock or prompt_manager.current_time_note()}\n{facts}"

    def _update_history(self, prompt_manager, already_reported: str | None) -> str:
        response_text = self._remove_thinking(self.dialog_manager.llm_response_raw)
        source = self.ctx.source
        location = self.ctx.location
        if not self.is_killed:
            if already_reported != self.ctx.user_input:
                prompt_manager.update_history(self.ctx.display_name, self.ctx.user_input, response_text, source, location,
                                              timestamp=self.turn_timestamp, time_note=self.time_note)
            else:
                prompt_manager.update_history(None, None, response_text)
            if self._cut_round_result:
                prompt_manager.add_command_result(self._cut_round_result)
        trace("")
        return response_text

    def _reply_owed(self) -> bool:
        """True, once per request, when the request is about to end with nothing said to the
        user although commands ran, and the last round's commands are not all ones whose
        effect is the answer (a light turned off, the music paused): the result of the round
        it then circles back with is SILENT_END_NOTE. A turn that ran no command stays
        silent — the LLM chose to say nothing (a reaction to a message)."""
        if self._silent_end_followed_up or self.ctx.test:
            return False
        if any(dm.llm_response.strip() for dm in self._rounds):
            return False
        ran = self._command_names()
        if not ran:
            return False
        cp = self.dialog_manager.command_processor
        last = [c.split("(", 1)[0] for c in cp.executed_commands]
        if last and all(cp.commands.is_self_evident(name) for name in last):
            trace(f"Request ends silently on {', '.join(last)} — the effect is the answer, no follow-up")
            return False
        self._silent_end_followed_up = True
        info(f"Request would end with nothing said to the user after {', '.join(ran)} — "
             f"one more round to reply")
        cp.command_reply = self.SILENT_END_NOTE
        cp.command_reply_contains_error = False
        return True

    def _cap_hit(self) -> str | None:
        """After a round the LLM ended at its output cap: the first time, the note it is told
        so with is added to the round's result (so the request circles back, a round that ran
        no command included, and the note stands in the history as a command result); the
        second time, returns the notice the request ends with. None for a round that ended on
        its own."""
        dm = self.dialog_manager
        if dm.ai.last_stop_reason != 'max_tokens':
            return None
        if self._cap_hit_followed_up:
            error("The LLM hit its output cap again on the round it was told about it — ending the request.")
            return self.CAP_HIT_NOTICE
        self._cap_hit_followed_up = True
        cap = dm.ai.output_cap()
        cap_text = f"the {cap}-token output cap" if cap else "its output cap"
        said = dm.llm_response.strip()
        if said:
            tail = said.splitlines()[-1][-self.CAP_HIT_TAIL_CHARS:]
            info(f"LLM hit {cap_text} mid-reply, after {len(said)} chars — one more round to continue")
            note_text = self.CAP_HIT_TEXT_NOTE.format(cap=cap_text, tail=tail)
        else:
            info(f"LLM hit {cap_text} while still thinking, nothing said — one more round to reply")
            note_text = self.CAP_HIT_THINKING_NOTE.format(cap=cap_text)
        dm.command_processor.add_note(note_text)
        return None

    def _command_names(self) -> list[str]:
        """The names of the commands every round of this request ran, in order."""
        return [c.split("(", 1)[0] for dm in self._rounds for c in dm.command_processor.executed_commands]

    def _report_turn(self):
        """The beta telemetry's assistant event: what every round told the user — the text
        left once the command processor took the thinking and the command blocks out, so no
        command argument goes with it. Like an observer it must never break a request."""
        if self.ctx.test:
            return
        try:
            said = "\n".join(text for dm in self._rounds if (text := dm.llm_response.strip()))
            Telemetry.get_instance().assistant_turn(said, self.ctx.source, self._command_names())
        except Exception as e:
            error(f"Telemetry report failed: {e}")

    def _wait_for_audio_and_notify_outposts(self):
        if not self.ctx.text_only and self.ctx.server_sound and self.playback:
            self.playback.finish_reply(self.dialog_manager.llm_response)

    def _save_to_database(self, query_key: str, response_text: str):
        if self.LLM_OUTAGE_HISTORY_MARKER.strip() in response_text:
            # A zero-command outage row would block CachedLLM's majority vote for this
            # query forever (any command-less row vetoes caching), so never store one.
            trace("Not storing outage reply in DB.")
            return
        if "false_positive(" not in response_text and not self.ctx.test:
            FloxyDatabase.get_instance().submit(query_key.strip(), "",
                                                response_text,
                                                self.dialog_manager.last_response_was_cached,
                                                source=self.ctx.source,
                                                location=self.ctx.location)
        else:
            trace("Not storing false positive in DB.")

    @staticmethod
    def _invalidate_cached_response_if_device_missing(old_dialog_manager, command_reply):
        """A cached response that commands a nonexistent device would replay the failure on
        every future cache hit — evict its backing rows the first time it errors. Only round 1
        can be cache-served (follow-up rounds have cache_query=None), so this fires at most
        once per request.
        TODO: _save_to_database still stores the final follow-up round's text under the
        original query as a future cache candidate; correction-only rows could later form a
        thinner majority than the intended scene. CacheAuditor is the safety net for now."""
        if not old_dialog_manager.last_response_was_cached:
            return
        if "could not find device" not in _format_command_reply(command_reply).lower():
            return
        from assistant.llm.CachedLLM import CachedLLM
        ai = old_dialog_manager.ai
        if isinstance(ai, CachedLLM):
            ai.invalidate_served_response("cached response referenced a nonexistent device")
        else:
            info("Cached response errored on a missing device but the LLM is not a CachedLLM "
                 "wrapper — nothing to invalidate.")

    def _handle_background_commands(self, prompt_manager) -> Generator[AssistantEvent, None, str | None]:
        user_input_already_reported_on = None
        round_num = 0
        unterminated_reports = 0
        previous_round: tuple | None = None
        repeats = 0
        while (self.dialog_manager.command_processor.command_running_in_background or
               self.dialog_manager.command_processor.command_reply is not None or
               self.dialog_manager.command_processor.unterminated_command_report is not None or
               not self.dialog_manager.command_processor.command_queue.empty() or
               self._reply_owed()):
            self._check_killed()
            round_num += 1

            old_dialog_manager = self.dialog_manager

            self._wait_for_background_command(old_dialog_manager)
            yield from self._drain_events()

            # Folded in only now: a command finishing earlier would have replaced it.
            report = old_dialog_manager.command_processor.take_unterminated_command_report()
            if report and unterminated_reports >= self.MAX_UNTERMINATED_COMMAND_REPORTS:
                error(f"Unterminated command again after {unterminated_reports} report(s) to the LLM "
                      f"this request — not circling back for it another time.")
            elif report:
                unterminated_reports += 1
                info(f"Reporting the unterminated command to the LLM ({unterminated_reports} of "
                     f"{self.MAX_UNTERMINATED_COMMAND_REPORTS} this request).")
                old_dialog_manager.command_processor.append_command_reply(report)

            if (old_dialog_manager.command_processor.command_reply is None and
                    old_dialog_manager.command_processor.command_queue.empty() and
                    not self._reply_owed()):
                info("Background command completed silently, skipping LLM follow-up.")
                break

            info("Receiving background command reply, circling back to LLM...")
            response_text = self._remove_thinking(self.dialog_manager.llm_response_raw)
            # pre_heat=False: the follow-up round's inference starts right below and
            # would contend with a pre-heat for Ollama's single slot; the terminal
            # _update_history after the loop fires the pre-heat instead.
            if user_input_already_reported_on != self.ctx.user_input:
                prompt_manager.update_history(self.ctx.display_name, self.ctx.user_input, response_text,
                                              self.ctx.source, self.ctx.location, pre_heat=False,
                                              timestamp=self.turn_timestamp, time_note=self.time_note)
                user_input_already_reported_on = self.ctx.user_input
            else:
                trace("User query already recorded for this request — appending assistant command turn only.")
                prompt_manager.update_history(None, None, response_text, pre_heat=False)

            self.dialog_manager = self._new_dialog_manager(self.ctx.test)
            self._rounds.append(self.dialog_manager)
            self.dialog_manager.select_ai(self.ctx)
            self.dialog_manager.select_voice(self.ctx)
            # A fresh dialog manager is fresh LLM instances: without the request's prefix
            # verdict the round would write its blocks at the long tier no later request reads.
            self._carry_per_request_prefix()
            self.dialog_manager.messages = old_dialog_manager.messages
            self.dialog_manager.reply_from_role = "user"
            # The follow-up round is the same byte-prefix request: folding directives onto the
            # result turn would put the divergence before the runner's restore point.
            self.dialog_manager.byte_prefix_tail = old_dialog_manager.byte_prefix_tail
            self.dialog_manager.request_id = old_dialog_manager.request_id
            if self.dialog_manager.request_id:
                trace(f"Carried request_id {self.dialog_manager.request_id} into command follow-up for text streaming")

            self._wait_for_command_reply(old_dialog_manager)
            yield from self._drain_events()

            command_reply = old_dialog_manager.command_processor.command_reply
            fine("command reply:\n" + _format_command_reply(command_reply))
            self._invalidate_cached_response_if_device_missing(old_dialog_manager, command_reply)
            executed = old_dialog_manager.command_processor.executed_commands
            this_round = (tuple(executed), command_reply) if isinstance(command_reply, str) else None
            repeats = repeats + 1 if this_round is not None and this_round == previous_round else 0
            previous_round = this_round
            if old_dialog_manager.command_processor.command_reply_contains_error:
                info(f"Command reply contains error(s) — framing follow-up as an error report "
                     f"({len(executed)} command(s) already executed).")
                self.dialog_manager.reply_to_llm = _build_error_round_reply(command_reply, executed)
            elif isinstance(command_reply, list):
                self.dialog_manager.reply_to_llm = command_reply
            elif repeats:
                info(f"Round {round_num} repeated the previous round's command(s) to the same result "
                     f"({repeats} of {self.MAX_REPEATED_ROUNDS}) — saying so under the result.")
                self.dialog_manager.reply_to_llm = _build_repeated_round_reply(command_reply)
            else:
                self.dialog_manager.reply_to_llm = f"result: \n{command_reply}"
            # A toolkit reference is stored whole: cut in half it is worse than none, and its
            # header would still count it as loaded. A round that batched a load_toolkit with
            # other commands (the prompt encourages batching) goes in uncapped too.
            loaded_toolkit = any(c.startswith(("load_toolkit(", "load_watch(")) for c in executed)
            if loaded_toolkit and len(executed) > 1:
                trace(f"Command result stored uncapped: load_toolkit ran alongside {len(executed) - 1} other command(s)")
            result_text = _command_result_for_history(self.dialog_manager.reply_to_llm)
            prompt_manager.add_command_result(result_text, cap=not loaded_toolkit)
            if repeats >= self.MAX_REPEATED_ROUNDS:
                error(f"Round {round_num} repeated the same command(s) to the same result for the {repeats}. time "
                      f"({', '.join(executed)}) — ending the request instead of circling back again.")
                yield Notice(self.REPEATED_ROUND_NOTICE)
                yield RoundEnd()
                return user_input_already_reported_on
            # The model gets the whole result, and so does every later round of this request;
            # the history keeps a cut copy of an oversized one, so no later request repeats the
            # tail. ClaudeLLM caches a follow-up round's tail for its short tier accordingly.
            self.dialog_manager.follow_up = True
            self.dialog_manager.use_vision_for_image_reply()
            try:
                yield from self.stream_response()
            except RequestKilledException:
                raise
            except Exception as e:
                # The response body is already open by now, so an escaping exception can
                # only sever the socket mid-turn — end politely instead, keeping history
                # bookkeeping for the rounds that completed.
                error(f"Stream error in command follow-up round {round_num}: {type(e).__name__}: {e}")
                yield Notice(self._llm_failure_notice(e))
                yield RoundEnd()
                return user_input_already_reported_on
            cut_notice = self._cap_hit()
            if cut_notice:
                yield Notice(cut_notice)
            yield RoundEnd(self.dialog_manager.llm_response_raw)
            self._wait_for_initial_command()
            yield from self._drain_events()
            if cut_notice:
                yield from self._settle_cut_round()
                return user_input_already_reported_on
            trace("Stream done.")
        return user_input_already_reported_on

    def _settle_cut_round(self) -> Generator[AssistantEvent, None, None]:
        """The round the request ends on after the second cut may have dispatched commands
        before it overran the cap: they run to the end here, so the request is still in
        progress while they do, and their result is kept for the history (_update_history
        persists it after the round's own text, where the next turn expects a command row) —
        the request just does not circle back with it."""
        dm = self.dialog_manager
        self._wait_for_background_command(dm)
        yield from self._drain_events()
        reply = dm.command_processor.command_reply
        if reply is None:
            trace("The cut round ran no command with a result — nothing to keep for the history")
            return
        executed = dm.command_processor.executed_commands
        info(f"The cut round's command(s) finished ({', '.join(executed) or 'none succeeded'}) — keeping the "
             f"result in the history without another round")
        self._cut_round_result = _command_result_for_history(reply if isinstance(reply, list) else f"result: \n{reply}")

    def generate_events(self, prompt_manager) -> Generator[AssistantEvent, None, None]:
        """The whole request as an event stream: the user's round, one round per command
        follow-up (each closed by RoundEnd), then Done — or Notice and an early end on
        denial / failure, and a silent end when killed."""
        try:
            try:
                self._rounds = [self.dialog_manager]
                if not self.ctx.test:
                    Telemetry.get_instance().user_turn(self.ctx.user_input, self.ctx.source)
                query_key = self._prepare_llm_context(prompt_manager)

                if query_key is None:
                    yield Notice("Request denied.")
                    return

                try:
                    yield from self.stream_response()
                except RequestKilledException:
                    raise
                except Exception as e:
                    error(f"Stream error: {type(e).__name__}: {e}")
                    yield Notice(self._llm_failure_notice(e))
                    return

                self._cap_hit()  # the first round is never the second cut: no notice to carry
                yield RoundEnd(self.dialog_manager.llm_response_raw)

                self._wait_for_initial_command()
                yield from self._drain_events()
                user_input_already_reported_on = yield from self._handle_background_commands(prompt_manager)

                if self.is_killed:
                    raise RequestKilledException()

                response_text = self._update_history(prompt_manager, user_input_already_reported_on)
                self._report_turn()
                self._wait_for_audio_and_notify_outposts()

                self._save_to_database(query_key, response_text)

                if self.ctx.server_sound and self.playback:
                    self.playback.mark_sound_played()

                trace("Request complete.")
                yield from self._drain_events()
                yield Done(self.dialog_manager.llm_response_raw)
            except RequestKilledException:
                info("Request killed mid-stream; ending the stream without Done.")
                return
        finally:
            self.is_performing_request = False
            self._close_text_stream(self.dialog_manager.request_id)
            self._notify_autonomous_user_ended()
            self._restore_outposts_if_muted()

    def _ensure_outposts_muted(self):
        """Mute the local mic outposts before the first sound of this reply — the thinking
        filler or the first audio chunk, whichever comes first. Applies to server-speaker AND
        client-streamed audio: a phone playing the reply in the same room would otherwise be
        transcribed by a local outpost. No-op without a voice interface."""
        if self.playback:
            self.playback.ensure_outposts_muted()

    def _restore_outposts_if_muted(self):
        """Failsafe unmute for a reply killed/errored mid-speech; see
        ServerPlayback.restore_outposts_if_muted."""
        if self.playback:
            self.playback.restore_outposts_if_muted()

    def _notify_autonomous_user_started(self):
        try:
            from assistant.agents.AutonomousActionManager import AutonomousActionManager
            aam = AutonomousActionManager._instance
            if aam is not None:
                aam.notify_user_interaction_started()
        except Exception as e:
            trace(f"AutonomousActionManager start-notify failed (non-fatal): {e}")

    def _notify_autonomous_user_ended(self):
        try:
            from assistant.agents.AutonomousActionManager import AutonomousActionManager
            aam = AutonomousActionManager._instance
            if aam is not None:
                aam.notify_user_interaction_ended()
        except Exception as e:
            trace(f"AutonomousActionManager end-notify failed (non-fatal): {e}")

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            if self.playback:
                self.playback.stop()
        except Exception:
            pass


class RequestData:
    def __init__(self, method, url, args, headers, json_body=None):
        self.method = method
        self.url = url
        self.args = args
        self.headers = headers
        self.json_body = json_body
