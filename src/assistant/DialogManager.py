import re
import queue
import threading
import time
from typing import Callable, Generator, TYPE_CHECKING

import httpx
from anthropic import APIConnectionError
from google.genai.errors import ServerError as GeminiServerError


from common.RequestKilledException import RequestKilledException
from assistant.commands.CommandProcessor import CommandProcessor
from assistant.commands.formats.CommandFormats import CommandFormats
from assistant.AssistantEvents import AssistantEvent, observers
from assistant.ChannelProfile import profile_for
from assistant.llm import create_llm, parse_llm_config, command_format_override, command_format_chooser, content_blocks_contain_images
from assistant.llm.AbstractLLM import LLMRefusalError
from assistant.llm.CachedLLM import CachedLLM
from assistant.prompt.MessageHistory import MessageHistory
from common.Language import language_code
from common.logging import info, trace, error, fine
from common.TTFTMonitor import TTFTMonitor

if TYPE_CHECKING:
    from interfaces.voice.tts.VoiceSynthesizer import VoiceSynthesizer


def strip_backticks(chunk: str, before: str, in_span: bool) -> tuple[str, bool]:
    """The chunk without the backticks a model wraps a name or a call in: they are not for
    the user, and the command processor reads the call without them. The one opening a
    span straight after a letter or digit becomes a space, so the words stay apart; every
    other one, the closing ones among them, is dropped, so the full stop after a span
    stays on its word (when every backtick became a space, `name`. reached the chat as
    "name ." — #143). *before* is the text the chunk continues, *in_span* whether a span
    was open at its end; both come from the stream so far, so a chunk cut inside a span
    reads the same as one that holds it whole."""
    out = []
    prev = before[-1:]
    for char in chunk:
        if char == "`":
            out.append(" " if not in_span and prev.isalnum() else "")
            in_span = not in_span
        else:
            out.append(char)
        prev = char
    return "".join(out), in_span


class DialogManager:
    """Runs one LLM round: picks the model and voice, streams the reply and hands commands off.

    Owns the fast/smart/vision LLM clients (cached outside test-suite mode) and the per-round
    CommandProcessor; the processed text is fed to TTS when the request has a voice.
    """
    # Set per request by RequestProcessor._prepare_llm_context: a byte-prefix request sends
    # nothing the next turn's history rendering would drop, so no trailing directives.
    byte_prefix_tail = False

    # Rough chars-per-token divisor for the live output-token estimate shown on the
    # Telegram "Thinking…" spinner. The <thinking> block is stripped before any token
    # reaches the client, so there's nothing to count downstream — but the raw stream is
    # accumulated in llm_response_raw, which gives a cheap, backend-agnostic estimate.
    CHARS_PER_TOKEN_ESTIMATE = 4
    _format_choices_logged: set[tuple[str, str]] = set()

    def __init__(self, web_request_handler, test_suite_mode: bool = False,
                 voice_factory: "Callable[[str | None], VoiceSynthesizer] | None" = None):
        """voice_factory builds the TTS synthesizer for a request language (see
        interfaces.voice.tts.VoiceFactory). None means no voice interface in this process:
        select_voice leaves self.voice None and every reply is text-only."""
        fast_llm_name, fast_model = parse_llm_config('fast')
        smart_llm_name, smart_model = parse_llm_config('smart')
        vision_llm_name, vision_model = parse_llm_config('vision')

        self.ai_fast = create_llm(fast_llm_name, fast_model, command_format=command_format_override('fast'))
        smart_format = command_format_override('smart')
        self.ai_smart = create_llm(smart_llm_name, smart_model, prompt_cache=True, command_format=smart_format)
        # The chat tier's model decides the syntax every reply is parsed in — before the cached
        # wrappers and the command processor below build their parsers from it. A `command_format:`
        # key in the tier's `ai.smart` block, or FLOXY_COMMAND_FORMAT, overrides the class's choice.
        self._command_format_source = command_format_chooser('smart') or f"the smart tier's {smart_llm_name} class"
        CommandFormats.set_active(self.ai_smart.command_format(), self._command_format_source)
        # Used whenever a request carries images — Opus 4.8 is much stronger at vision
        # than the 4.6 we run as smart. See select_ai (images always win).
        self.ai_vision = create_llm(vision_llm_name, vision_model, prompt_cache=True,
                                    command_format=command_format_override('vision'))

        TTFTMonitor.get_instance().register_models(
            smart_model=self.ai_smart.model, fast_model=self.ai_fast.model
        )

        if not test_suite_mode:
            self.ai_fast = CachedLLM(self.ai_fast)
            self.ai_smart = CachedLLM(self.ai_smart)
            self.ai_vision = CachedLLM(self.ai_vision)

        self.ai = self.ai_fast  # default to fast (voice)

        self.command_processor = CommandProcessor(test_suite_mode)

        for ai in (self.ai_fast, self.ai_smart, self.ai_vision):
            if isinstance(ai, CachedLLM):
                ai.update_command_list(self.command_processor.possible_commands)
        choice = (CommandFormats.active().name, self._command_format_source)
        if choice not in DialogManager._format_choices_logged:   # once per process, not per pool slot
            DialogManager._format_choices_logged.add(choice)
            info(f"Command format: {choice[0]}, chosen by {choice[1]} — "
                 f"the syntax every command in a reply is parsed in")

        self.llm_response = ""
        self.llm_response_raw = ""
        self.time_to_first_token = 0
        self.time_to_user_reply = 0
        self.messages = []
        self.first_token_got_at = 0
        self.reply_token_got_at = 0
        self.test_suite_mode = test_suite_mode
        self.is_killed = False
        self.reply_to_llm: str | list = ""
        self.reply_from_role = "user"
        # A command follow-up round: its tail is the result the next round of this request
        # repeats verbatim and no later request does. ClaudeLLM caches it for the short tier.
        self.follow_up = False
        # Clean user utterance used as the response-cache key. Set by the request handler
        # for genuine user turns only; stays None for image and command-follow-up turns so
        # those bypass the cache. See CachedLLM.create_streaming_completion.
        self.cache_query: str | None = None

        self.voice_factory = voice_factory
        # One synthesizer per request language for the life of this dialog, so a warm
        # backend connection survives across turns; all closed on kill.
        self._voices: "dict[str | None, VoiceSynthesizer]" = {}
        self.voice: "VoiceSynthesizer | None" = None  # set per request by select_voice

        self.last_response_was_cached = False
        self.web_request_handler = web_request_handler
        # In-stream events (Thinking, CommandRan, ...) share the handler's side channel so
        # they surface in the request's event stream; a lone dialog gets its own queue.
        handler_events = getattr(web_request_handler, "events", None)
        self.events = handler_events if isinstance(handler_events, queue.SimpleQueue) else queue.SimpleQueue()
        self.thinking = False  # Whether to use native thinking for this request
        self.request_id: str | None = None  # Correlates this request with the /stream_text endpoint

    # Runs on: request thread (called from CommandProcessor the moment a
    # <thinking> block opens during token streaming).
    def notify_thinking_started(self):
        """Play a short filler ("Hmm...", "Let's see...") to cover the silent gap while
        the model emits its <thinking> tokens. Server speakers only, and only when the
        handler has a server playback (i.e. a voice interface is running) — the playback
        mutes the mic outposts before the filler sounds, so ordering stays synchronous."""
        try:
            ctx = self.web_request_handler.ctx
        except AttributeError:
            return
        if not getattr(ctx, "server_sound", False):
            return
        playback = getattr(self.web_request_handler, "playback", None)
        if playback is None:
            info("No server playback in this process — thinking filler skipped.")
            return
        # Use the request's language (Whisper for local outposts, the client for apps)
        # so the filler matches the reply language; the playback falls back to the
        # server's configured default when unknown.
        try:
            playback.play_thinking_filler(getattr(ctx, "lang", None))
        except Exception as e:
            fine(f"Failed to play thinking filler: {e}")

    def select_ai(self, ctx):
        if ctx.images:
            self.ai = self.ai_vision
            info(f"Request carries {len(ctx.images)} image(s) — using vision tier ({self.ai.name()})")
        elif ctx.ai_override == "fast":
            self.ai = self.ai_fast
        elif ctx.ai_override == "smart":
            self.ai = self.ai_smart
        else:
            profile = profile_for(ctx.source)
            self.ai = self.ai_fast if profile.llm_tier == "fast" else self.ai_smart
            trace(f"LLM tier for {profile.name}: {profile.llm_tier} ({self.ai.name()})")

    def use_vision_for_image_reply(self):
        """Switch to the vision tier when the pending reply_to_llm carries image blocks.
        select_ai only sees ctx.images (set on the user's own turn); command replies —
        e.g. camera snapshots — arrive in reply_to_llm later, after select_ai already ran."""
        if not content_blocks_contain_images(self.reply_to_llm):
            return
        self.ai = self.ai_vision
        info(f"Command reply carries images — using vision tier ({self.ai.name()})")
        # The <thinking>-tag instruction was decided for the tier the request started on and
        # rides along in self.messages. A vision model that thinks on its own (Claude) refuses
        # the whole round over it, and so would its refusal fallback.
        if self.ai.handles_thinking_internally():
            stripped = MessageHistory.strip_thinking_instruction(self.messages)
            if stripped:
                info(f"{self.ai.name()} thinks on its own — <thinking> tag instruction removed from {stripped} turn(s)")
            else:
                trace(f"{self.ai.name()} thinks on its own — the messages carry no <thinking> tag instruction")

    def _voice_for_lang(self, lang: str | None) -> "VoiceSynthesizer | None":
        if self.voice_factory is None:
            return None
        voice = self._voices.get(lang)
        if voice is None:
            voice = self.voice_factory(lang)
            if voice is not None:   # no engine can speak the language: text-only, and asked again next time
                self._voices[lang] = voice
        return voice

    def select_voice(self, ctx):
        self.voice = self._voice_for_lang(ctx.lang)
        if self.voice is None:
            info(f"No voice factory — request for lang={ctx.lang} runs text-only.")
            return
        self.voice.set_request_language(ctx.lang)
        # The night voice is for the house speakers; a client playing the reply itself (the apps) never whispers.
        self.voice.set_whisper_allowed(ctx.server_sound)
        trace(f"Voice for lang={ctx.lang}: {type(self.voice).__name__}, "
             f"night voice {'allowed' if ctx.server_sound else 'off (the client plays the reply)'}")

    # Per-request reply-language directive, written in the target language. Folded onto the
    # end of the request (see stream_llm_response / _with_trailing_directives) so it overrides
    # the otherwise English-dominated context. It uses the language we already detected
    # (ctx.lang — Whisper for local outposts, the client for apps) instead of making the
    # model infer it from a short transcript, which it got wrong. Phrased in-language so
    # it doubles as an example of the language to answer in.
    _LANGUAGE_DIRECTIVES = {
        "sv": "Användaren pratar svenska med dig. Använd svenska när du svarar användaren.",
        "en": "The user is speaking English with you. Reply to the user in English.",
    }

    # Voice-to-voice replies must stay terse. The static prompt rule for this is easy to
    # miss mid-context (Gemini especially runs long), so we also fold it onto the very end
    # of the request — the highest-attention spot. Text/chat replies aren't length-capped.
    _BREVITY_DIRECTIVE = "Make sure your response is under 20 words!"

    def _language_directive(self) -> str | None:
        lang = getattr(getattr(self.web_request_handler, "ctx", None), "lang", None)
        if not isinstance(lang, str):
            lang = None
        code = language_code(lang)
        directive = self._LANGUAGE_DIRECTIVES.get(code)
        if not directive:
            fine(f"No reply-language directive for '{code}' — leaving the language to the model.")
        return directive

    def _brevity_directive(self) -> str | None:
        ctx = getattr(self.web_request_handler, "ctx", None)
        if not profile_for(getattr(ctx, "source", None)).brevity_directive:
            return None
        return self._BREVITY_DIRECTIVE

    def _trailing_directives(self) -> list[str]:
        """Per-request directives folded onto the end of the request, in read order:
        brevity first, language last so the reply-language directive stays the final thing
        the model reads (it must override the English-dominated context — see above)."""
        return [d for d in (self._brevity_directive(), self._language_directive()) if d]

    def _with_trailing_directives(self, messages: list) -> list:
        """Return a copy of `messages` with the per-request trailing directives folded into
        the final user turn. Ephemeral on purpose — they ride along to the LLM but are never
        stored in self.messages, so each turn carries its own current directives and nothing
        accumulates in history. Folds into the existing user turn (text, or a text block for
        image/content-block messages) rather than appending a new message, so it can't create
        the consecutive same-role turns Claude/Gemini reject."""
        if self.byte_prefix_tail:
            trace("Byte-prefix tail: no trailing directives folded into the request")
            return messages
        directives = self._trailing_directives()
        if not directives or not messages:
            return messages
        directive = "\n\n".join(directives)
        result = list(messages)
        last = result[-1]
        content = last.get("content")
        if last.get("role") == self.reply_from_role and isinstance(content, str):
            result[-1] = {**last, "content": content + "\n\n" + directive}
        elif last.get("role") == self.reply_from_role and isinstance(content, list):
            result[-1] = {**last, "content": content + [{"type": "text", "text": directive}]}
        else:
            result.append({"role": self.reply_from_role, "content": directive})
        fine(f"Trailing directives folded into the request: {directive!r}")
        return result

    @property
    def reply_text(self) -> str:
        if isinstance(self.reply_to_llm, str):
            return self.reply_to_llm
        return " ".join(
            b.get("text", "") for b in self.reply_to_llm
            if isinstance(b, dict) and b.get("type") == "text"
        )

    def kill(self):
        fine("Dialog manager is being killed.")
        self.command_processor.kill()
        self.is_killed = True
        # A streaming call still running stops re-sending: a superseded request kept making
        # full Opus calls for 80 s after its kill (2026-10-23, issue #108).
        self.ai.cancel()

        voices = list(self._voices.values())
        if not voices:
            trace("No voices to close.")
            return

        def close_voice_safely():
            for voice in voices:
                try:
                    voice.close()
                except Exception as e:
                    trace(f"Exception closing voice (non-critical): {e}")

        threading.Thread(target=close_voice_safely, daemon=True).start()

    def emit(self, event: AssistantEvent):
        """Queue an in-stream event (any thread); drained at the next yield point."""
        self._event_queue().put(event)

    def drain_events(self) -> Generator[AssistantEvent, None, None]:
        q = self._event_queue()
        while True:
            try:
                yield q.get_nowait()
            except queue.Empty:
                return

    def _event_queue(self) -> "queue.SimpleQueue[AssistantEvent]":
        q = self.__dict__.get("events")
        if q is None:
            q = self.events = queue.SimpleQueue()
        return q

    def current_output_token_estimate(self) -> int:
        """Live estimate of output tokens produced so far this request. Drives the token
        count on the Telegram "Thinking…" spinner; resets per request since a fresh
        DialogManager (and thus llm_response_raw) is built for each one."""
        return len(self.llm_response_raw) // self.CHARS_PER_TOKEN_ESTIMATE

    # The assistant turn that closes the test suite's toolkit result when the query is sent as
    # a turn of its own (_query_as_own_turn): a fixture like the toolkit turn before it.
    TEST_SUITE_TOOLKITS_ACK = "Toolkits loaded."

    def _query_as_own_turn(self) -> bool:
        """In the test suite, on a backend that asks for it (AbstractLLM.prefers_query_as_own_turn),
        the query follows the toolkit turn as a user turn of its own — behind a short assistant
        turn, since a user turn right after a tool result is rendered into the same block."""
        ai = getattr(self, "ai", None)
        return bool(self.test_suite_mode and ai is not None and ai.prefers_query_as_own_turn())

    def stream_llm_response(self) -> Generator[str, None, None]:
        if (self.messages and self.messages[-1]["role"] == self.reply_from_role
                and isinstance(self.reply_to_llm, str) and isinstance(self.messages[-1]["content"], str)
                and self._query_as_own_turn()):
            trace("Test suite: the query is sent as its own user turn after the toolkit result")
            self.messages.append({"role": "assistant", "content": self.TEST_SUITE_TOOLKITS_ACK})
            self.messages.append({"role": self.reply_from_role, "content": self.reply_to_llm})
        elif (self.messages and self.messages[-1]["role"] == self.reply_from_role
                and isinstance(self.reply_to_llm, str) and isinstance(self.messages[-1]["content"], str)):
            # Replace, never mutate: the list is a shallow copy, so the trailing dict may be
            # shared with another holder (the test-suite fixtures in PromptManager) and an
            # in-place += would leak this query into every later request.
            last = self.messages[-1]
            self.messages[-1] = {**last, "content": last["content"] + "\n\n" + self.reply_to_llm}
        else:
            self.messages.append({"role": self.reply_from_role, "content": self.reply_to_llm})

        messages = self._with_trailing_directives(self.messages)

        from assistant.prompt.PromptManager import PromptManager
        PromptManager.save_prompt_to_file(messages)

        if self.is_killed:
            raise RequestKilledException()

        self.ai.initialize()

        if self.is_killed:
            raise RequestKilledException()
        info(f"Sending to {self.ai.name()}...")
        self.command_processor.clear_command_reply()
        t = time.time()
        time_shown = False

        from assistant.llm.CachedLLM import CachedLLM
        if isinstance(self.ai, CachedLLM):
            self.ai.cache_query = self.cache_query

        yield from self._stream_with_transport_retry(messages, t, time_shown)

        # Track if response was from cache (only for CachedLLM)
        from assistant.llm.CachedLLM import CachedLLM
        if isinstance(self.ai, CachedLLM):
            self.last_response_was_cached = self.ai.last_response_was_cached
        else:
            self.last_response_was_cached = False

        # The turn as the history keeps it: a call the model wrote in a foreign syntax and the
        # processor rescued is written in the active one, so the model never sees that shape
        # in its own past turns (the feedback loop that made Opus repeat it).
        self.llm_response_raw = self.command_processor.format.rewritten(
            self.command_processor.without_dropped_thinking(self.llm_response_raw))
        self.messages.append({"role": "assistant", "content": self._turn_for_messages()})

    # What stands for an assistant turn that produced no text — the output cap spent on
    # thinking (native, so not in the stream) — when the request circles back on it: the
    # Messages API rejects an empty or whitespace-only non-final message with a 400.
    EMPTY_TURN_PLACEHOLDER = "[no reply was produced in this round]"

    def _turn_for_messages(self) -> str:
        if self.llm_response_raw.strip():
            return self.llm_response_raw
        info("The round produced no text — a placeholder stands for it in the request's messages")
        return self.EMPTY_TURN_PLACEHOLDER

    STREAM_DROP_RETRIES = 2
    STREAM_DROP_RETRY_DELAY = 1.0
    STREAM_DROP_HISTORY_MARKER = "\n[reply interrupted by connection drop]"

    def _surface_notice(self, notice: str):
        """Make an out-of-band notice (refusal, transport drop) reach the user.

        Notices yielded outside process_llm_stream skip the first-token setup, so
        nothing has started audio playback — the queue-drain thread only starts via
        initialize_audio_playback(), and without it TTS chunks pile into audio_queue
        with no consumer and the notice plays as silence on server-sound requests.
        The /stream_text overlay publish is likewise normally done per-token inside
        process_llm_stream, so do it here too."""
        if not self.web_request_handler.ctx.text_only:
            self.web_request_handler.initialize_audio_playback()
        self.web_request_handler.publish_text_event(getattr(self, "request_id", None), {"type": "token", "text": notice})

    def _stream_with_transport_retry(self, messages: list, t: float, time_shown: bool) -> Generator[str, None, None]:
        """Survive the LLM backend dropping the connection mid-stream.

        Transparent retry is only safe while nothing user-facing has happened: no reply
        text emitted and no command dispatched. That covers the most probable drop
        window — long thinking blocks are exactly the streams that live long enough to
        get cut. (ClaudeLLM already retries drops before its first delta; this layer is
        the only one that knows whether yielded deltas were all swallowed thinking text.)

        Once output or side effects exist, regeneration is non-deterministic and could
        re-fire commands, so degrade gracefully instead: yield a short truncation notice
        so the response body completes cleanly, and keep the partial turn (marked as
        interrupted) so the caller's history append preserves it and "continue" works.

        A safety refusal (LLMRefusalError, already retried in the LLM layer) yields no
        text and fires no commands, so it is handed to the LLM's refusal_fallback() once
        — Fable's classifier refuses deterministically, Opus answers — with a trailing
        notice telling the user which model spoke; without a fallback, or when the
        fallback refuses too, a refusal notice is the reply.
        """
        raw_snapshot = self.llm_response_raw
        original_ai = self.ai
        try:
            yield from self._stream_with_retry_and_fallback(messages, t, time_shown, raw_snapshot)
        finally:
            self.ai = original_ai

    def _stream_with_retry_and_fallback(self, messages: list, t: float, time_shown: bool,
                                        raw_snapshot: str) -> Generator[str, None, None]:
        refused_ai = None
        attempt = 0
        while True:
            try:
                response = self.ai.create_streaming_completion(messages, thinking=self.thinking,
                                                               follow_up=self.follow_up)
                yield from self.process_llm_stream(response, t, time_shown)
                if refused_ai is not None:
                    yield from self._surface_fallback_notice(refused_ai)
                return
            except LLMRefusalError:
                fallback = self.ai.refusal_fallback() if refused_ai is None else None
                if fallback is None:
                    yield from self._surface_refusal_notice()
                    return
                info(f"{self.ai.name()} refused the request (safety filter) — "
                     f"falling back to {fallback.name()} for this request.")
                refused_ai, self.ai = self.ai, fallback
                self.ai.initialize()
            except (httpx.TransportError, APIConnectionError, GeminiServerError) as e:
                # a Gemini 5xx before the first token is as transient as a dropped socket (2026-10-19:
                # one "500 INTERNAL" two seconds in cost a voice request its reply)
                had_output = bool(self.llm_response)
                commands_fired = self.command_processor.commands_dispatched
                if had_output or commands_fired or attempt >= self.STREAM_DROP_RETRIES:
                    error(f"LLM stream dropped ({type(e).__name__}: {e}) after {len(self.llm_response)} reply chars "
                          f"and {commands_fired} dispatched commands — not retryable, truncating gracefully.")
                    notice = (" — the connection to the AI dropped mid-reply, that's all I got."
                              if had_output else
                              "The connection to the AI dropped. Please try again in a moment.")
                    self.llm_response += notice
                    self.llm_response_raw += self.STREAM_DROP_HISTORY_MARKER
                    self._surface_notice(notice)
                    yield notice
                    return
                info(f"LLM stream dropped with no user-facing output ({type(e).__name__}, "
                     f"attempt {attempt + 1}/{self.STREAM_DROP_RETRIES + 1}) — retrying transparently.")
                self.llm_response_raw = raw_snapshot
                self.command_processor.reset_parse_state()
                attempt += 1
                time.sleep(self.STREAM_DROP_RETRY_DELAY)

    def _surface_refusal_notice(self) -> Generator[str, None, None]:
        """A refusal yields no text and fires no commands, so a notice as the reply is
        safe on every client. name() can carry a model suffix ("Claude (claude-fable-5)")
        — too clunky for a spoken notice, so keep only the provider part."""
        ai_name = self.ai.name().split(" (")[0]
        notice = (f"{ai_name} refused to process this request due to security concerns. "
                  f"Rephrasing or asking again usually helps.")
        info("LLM refused the request (safety filter) — surfacing refusal notice to the user.")
        self.llm_response += notice
        self.llm_response_raw += "[the model's safety filter refused to answer this request]"
        self._surface_notice(notice)
        yield notice

    def _surface_fallback_notice(self, refused_ai) -> Generator[str, None, None]:
        """After the fallback's reply: tell the user who answered and that the switch was
        for this turn only. Not part of the raw turn — the history stays the model's own
        words — so the next request's prompt does not carry the refusal either."""
        notice = (f"\n\n({refused_ai.display_model()} refused this request, "
                  f"had to use {self.ai.display_model()} fallback)")
        info(f"Fallback reply complete — appending the {self.ai.display_model()}-answered notice.")
        self.llm_response += notice
        self._surface_notice(notice)
        yield notice

    def _leave_stream_at_the_cut(self, response):
        """The processor stopped reading the reply (its command cap, or a results block the
        model wrote itself): close the stream so the model stops generating, and keep of the
        turn exactly the calls that ran — the text up to the last of them, its block closed —
        so the follow-up round reads a reply of that many commands and nothing says there
        were more."""
        self._keep_the_cut_turn("stream closed")
        close = getattr(response, "close", None)
        if close is not None:
            close()

    def _keep_the_cut_turn(self, how: str):
        """Store of the cut reply exactly the calls that ran (CommandProcessor.kept_of)."""
        ran = self.command_processor.commands_dispatched
        before = len(self.llm_response_raw)
        self.llm_response_raw = self.command_processor.kept_of(self.llm_response_raw)
        info(f"Reply cut after {ran} commands — {how}, the turn kept as those calls "
             f"({before} → {len(self.llm_response_raw)} chars received)")

    def process_llm_stream(self, response, t: float, time_shown: bool) -> Generator[str, None, None]:
        # Emotion tags ([laughing], [sighing], ...) are synthesis directives, not display
        # text, so strip them before publishing to /stream_text or they render literally in
        # the iOS chat overlay. A tag can span chunks, so hold back a trailing unclosed '['.
        stream_text_buffer = ""
        reply_time_shown = False
        in_span = False   # inside a backtick span of the model's (strip_backticks)

        def publish_stream_text(chunk: str, flush: bool = False):
            nonlocal stream_text_buffer
            stream_text_buffer += chunk
            hold = 0
            if not flush:
                open_idx = stream_text_buffer.rfind("[")
                if open_idx != -1 and "]" not in stream_text_buffer[open_idx:]:
                    hold = len(stream_text_buffer) - open_idx
            safe = stream_text_buffer[:len(stream_text_buffer) - hold] if hold else stream_text_buffer
            stream_text_buffer = stream_text_buffer[len(stream_text_buffer) - hold:] if hold else ""
            if "[" in safe:
                safe = re.sub(r"\s*\[[^\]]*\]\s*", " ", safe)
            if safe:
                self.web_request_handler.publish_text_event(getattr(self, "request_id", None), {"type": "token", "text": safe})

        for content in response:
            if self.is_killed:
                raise RequestKilledException()

            if not time_shown:
                time_shown = True
                self.first_token_got_at = time.time()
                # Fallback: if the reply has no spoken text (pure command response) the
                # reply-start timestamp never updates, so default it to the first token —
                # voice synthesis then measures from here, as it did before this metric.
                self.reply_token_got_at = self.first_token_got_at
                self.time_to_first_token = round(self.first_token_got_at - t, 2)
                info(f"First token from {self.ai.name()}: {self.time_to_first_token}s.")
                # CachedLLM decides hit/miss before yielding its first token, so the flag
                # is already accurate here.
                from_cache = getattr(self.ai, "last_response_was_cached", False)
                observers().on_first_token(self.time_to_first_token, from_cache)
                # The dashboard's TTFT is the model's, whichever backend it is; a reply served
                # from the response cache never reached a model and would chart as 0 s.
                if not from_cache:
                    TTFTMonitor.get_instance().record(getattr(self.ai, "model", ""), self.time_to_first_token)
                # initialize audio playback as soon as possible, so pyaudio has time to boot up when we're waiting for first sound chunk
                if not self.web_request_handler.ctx.text_only:
                    self.web_request_handler.initialize_audio_playback()

            before = self.llm_response_raw[-1:]
            self.llm_response_raw += content
            content, in_span = strip_backticks(content, before, in_span)

            processed_chunk = self.command_processor.process(content, self)
            # Thinking chunks arrive while the processor swallows the block and yields no
            # text — surfacing them here keeps them live for the requesting interface.
            yield from self.drain_events()
            if processed_chunk is not None:
                reply_not_started = not self.llm_response or len(self.llm_response) <= 0
                if reply_not_started:
                    processed_chunk = processed_chunk.lstrip()  # strip whitespace at the beginning of response
                # First user-facing reply token — anything emitted after the optional
                # <thinking> block (stripped by the command processor). Measured from the
                # first token, so this row shows the thinking duration.
                if reply_not_started and processed_chunk and not reply_time_shown:
                    reply_time_shown = True
                    self.reply_token_got_at = time.time()
                    self.time_to_user_reply = round(self.reply_token_got_at - self.first_token_got_at, 2)
                    info(f"First reply token from {self.ai.name()}: {self.time_to_user_reply}s after first token (thinking).")
                    observers().on_reply_started(self.time_to_user_reply)
                self.llm_response += processed_chunk
                observers().on_tokens(processed_chunk)
                publish_stream_text(processed_chunk)
                yield processed_chunk

            if getattr(self.command_processor, "reply_cut", False):
                self._leave_stream_at_the_cut(response)
                break

        cut_in_stream = getattr(self.command_processor, "reply_cut", False)
        flushed = self.command_processor.flush_buffer(self)
        yield from self.drain_events()
        # The flush parses what the stream held back (the post-call hold, a call split across
        # chunks); a cut it raises there would otherwise leave the calls that never ran in the turn.
        if not cut_in_stream and getattr(self.command_processor, "reply_cut", False):
            self._keep_the_cut_turn("at the end of the stream")
        if flushed:
            self.llm_response += flushed
            observers().on_tokens(flushed)
            publish_stream_text(flushed)
            yield flushed

        publish_stream_text("", flush=True)
