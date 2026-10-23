import time
import traceback
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Generator

from common.logging import error, trace
from assistant.commands.formats.CommandFormat import CommandFormat
from assistant.commands.formats.KeepFormat import KeepFormat


class Thinking(IntEnum):
    """How much a request may reason, in the caller's terms — each backend maps it to what it
    has. NORMAL is a conversation turn: whatever the backend does by default. FULL is the
    memory curator: the deepest reasoning and the headroom for a long, complete answer (on
    Claude the adaptive-thinking parameter and the 64k output cap, on ds4 effort high, on
    Gemini thinking level high; ChatGPT has nothing for it and refuses). MORE is between —
    the autonomous check: more reasoning where a backend has levels to give (ds4 one step up,
    medium at least) and exactly NORMAL everywhere else, never the long-answer headroom. A
    plain bool is still accepted: True is FULL, False is NORMAL."""
    NORMAL = 0
    MORE = 1
    FULL = 2

    @classmethod
    def of(cls, value: "Thinking | bool") -> "Thinking":
        if isinstance(value, cls):
            return value
        return cls.FULL if value else cls.NORMAL

    @property
    def full(self) -> bool:
        return self is Thinking.FULL


class ThinkingNotSupportedError(Exception):
    """Raised when thinking mode is requested but not supported by this LLM."""
    pass


class LLMRefusalError(Exception):
    """Raised when the model's safety layer refused to answer (e.g. Anthropic
    stop_reason=refusal) and retries were exhausted."""
    pass


class LLMPreempted(Exception):
    """A background request (photo indexing) was aborted because an interactive request
    arrived for the same single-slot backend. Not a failure: the caller resends it once the
    backend is idle again."""
    pass


class LLMUnavailable(Exception):
    """The provider is refusing every request for now — the prepaid credits are gone (402) or a
    spend or rate ceiling is hit (429) — so the work is not failed, only not possible yet. Once
    the backend knows, it raises this without making the call, so thirty workers do not each
    discover it three times; `_retry` does not retry it, and a caller indexing a backlog leaves
    the item in its queue instead of writing it off. `unavailable()` says why, `probe()` whether
    requests go through again."""
    pass


def api_error_message(e: Exception) -> str | None:
    """The server's own explanation of a failed API call, for showing to the user —
    "Your credit balance is too low…", "Overloaded" — or None when the failure never
    reached a server (transport drop, timeout, a bug of ours).

    Duck-typed so the SDKs need not be imported here: the Anthropic and OpenAI status
    errors carry `status_code` and the decoded `body` (`{"error": {"message": …}}`), the
    Gemini one carries `code` and `message`."""
    status = getattr(e, "status_code", None) or getattr(e, "code", None)
    if not isinstance(status, int):
        return None
    body = getattr(e, "body", None)
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        message = body["error"].get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    message = getattr(e, "message", None)
    if isinstance(message, str) and message.strip():
        return message.strip()
    return f"HTTP {status}"


class AbstractLLM(ABC):
    """Defines the contract every LLM backend implements, plus shared retry and cache-marker helpers.

    CACHE_BREAKPOINT_MARKER is embedded in res/prompt.md; ClaudeLLM turns it into Anthropic
    cache_control breakpoints while every other backend strips it via strip_cache_markers().
    """

    # Prompt-cache breakpoint marker embedded in res/prompt.md. ClaudeLLM consumes
    # these to place Anthropic cache_control breakpoints; every other backend must
    # strip them (see strip_cache_markers) so they don't reach the model as text.
    CACHE_BREAKPOINT_MARKER = "<<<CACHE_BREAKPOINT>>>"

    @classmethod
    def strip_cache_markers(cls, text):
        """Remove cache breakpoint markers from prompt text. No-op for non-str
        content (e.g. content-block lists, which never carry markers)."""
        if isinstance(text, str):
            return text.replace(cls.CACHE_BREAKPOINT_MARKER, "")
        return text

    def _ensure_initialized(self):
        """Verify the client has been initialized."""
        if not self.client:
            raise RuntimeError("Client not initialized. Call initialize() first.")

    def _retry(self, fn, max_retries=3):
        """Retry a callable up to max_retries times, with 1s sleep between attempts."""
        for attempt in range(max_retries):
            try:
                return fn()
            except (LLMRefusalError, LLMPreempted, LLMUnavailable):
                raise  # deterministic / the caller resends when the backend is idle — no retry here
            except Exception as e:
                if attempt == max_retries - 1:
                    error(f"Failed after {max_retries} attempts")
                    trace(traceback.format_exc())
                    raise
                else:
                    trace(f"Failed to contact AI: {e}, attempt  {attempt + 1}.")
                    time.sleep(1)
        return "Could not get response from AI"

    def prefers_prepended_user_query(self) -> bool:
        """Returns True if this LLM prefers the user query prepended to the system prompt. Override in subclasses."""
        return True

    # The format a tier's `command_format:` config key named, set by `use_command_format`; a
    # class attribute so a backend that never calls this constructor still has it.
    _command_format_override: type[CommandFormat] | None = None

    def command_format(self) -> type[CommandFormat]:
        """The call syntax this instance writes commands in (see `assistant.commands.formats`):
        the one its tier's config named with `command_format:`, else the class's own."""
        return self._command_format_override or self.default_command_format()

    def default_command_format(self) -> type[CommandFormat]:
        """The call syntax this backend's models write most reliably: Keep's own unless the
        model family was trained on another. Override in subclasses."""
        return KeepFormat

    def use_command_format(self, fmt: type[CommandFormat] | None) -> None:
        """Make `fmt` what `command_format()` returns (None: back to the class's own) — the
        tier's `command_format:` override, applied by `create_llm` after construction."""
        self._command_format_override = fmt

    def set_prompt_cache(self, enabled: bool):
        """Enable/disable prompt caching for subsequent requests. Default: no-op.
        Override in subclasses that pay for caching (e.g. ClaudeLLM)."""
        pass

    def set_per_request_prefix(self, per_request: bool):
        """Tell the LLM that the prompt prefix of the coming request is shared by no later
        request (the user query is prepended to the top of the prompt): a prompt cache then
        serves only the request's own follow-up rounds. Default: no-op. Override in subclasses
        that pick a cache tier by it (ClaudeLLM)."""
        pass

    def refuses_reasoning_requests(self) -> bool:
        """True when the backend's safeguards stop a request that asks the model to write out
        its reasoning — as a JSON field, a tag, a "think step by step" line — so that the whole
        reply is lost (Anthropic's `reasoning_extraction` refusal). A prompt builder then leaves
        the reasoning field out for this backend instead of losing every reply."""
        return False

    def supports_native_thinking(self) -> bool:
        """Returns True if this LLM supports native thinking mode. Override in subclasses."""
        return False

    def handles_thinking_internally(self) -> bool:
        """True when the model reasons on its own, server-side, whether or not we ask: no
        prompt may then instruct it to write its reasoning into a <thinking> tag. Anthropic's
        classifier reads that instruction as reasoning extraction and refuses the whole
        request (stop_reason=refusal, category reasoning_extraction) — on Fable 5.1 and
        Opus 5 alike, so the fallback cannot help. Override in subclasses."""
        return False

    def prefers_byte_prefix_tail(self) -> bool:
        """True when the backend's cache restores only at the end of the previous request, so
        every request must be a byte prefix of the next one: the trailing turn is sent exactly
        as the history will render it and carries no per-request extras (see
        DevelopmentConfig.BYTE_PREFIX_TAIL). Override in subclasses."""
        return False

    def prefers_query_as_own_turn(self) -> bool:
        """True when the test suite's query should be its own user turn, behind a short assistant
        turn that closes the toolkit result, instead of being merged into that result: ds4-server
        cuts its reusable disk checkpoint right before a prompt's last user turn
        (`--kv-cache-cold-anchor last-user`), so with the query apart every test loads the whole
        fixed prefix and prefills the query alone. Production requests are not affected — the
        merge stays, Claude and Gemini reject same-role adjacency. Override in subclasses."""
        return False

    def pre_heat_cache(self, messages: list[dict]):
        """Pre-heat the LLM's cache with the given messages. Default: no-op.
        Override in subclasses that benefit from cache warming (e.g., local LLMs)."""
        pass

    @property
    def last_stop_reason(self) -> str | None:
        """Why the most recent completion ended, in the provider's own words ('end_turn',
        'max_tokens', 'refusal'…), or None when the backend does not report it. A caller
        that persists a reply verbatim checks this for 'max_tokens' before trusting it."""
        return getattr(self, '_last_stop_reason', None)

    def output_cap(self) -> int | None:
        """The output-token cap a chat request is sent with, named in the note the request
        circles back with when a round hits it (RequestProcessor), or None when the backend
        has no fixed cap."""
        return None

    def cancel(self):
        """The request this LLM serves was killed: a streaming call still running stops
        re-sending (an empty reply's retry) and returns. Default: nothing to stop."""
        pass

    def think(self, system_message: str, user_message: str, images: list | None = None, thinking: bool = False,
              background: bool = False) -> str:
        """
        Sends a request to the LLM, blocks until done, and returns the response.
        `background` marks bulk work (photo indexing) that a single-slot backend runs only while
        idle and aborts (LLMPreempted) when a user request arrives; cloud backends ignore it.
        """
        def fn():
            if images:
                return self.create_completion_with_images(
                    [{'role': 'system', 'content': system_message}, {'role': 'user', 'content': user_message}],
                    images, background=background)
            else:
                return self.create_completion(
                    [{'role': 'system', 'content': system_message}, {'role': 'user', 'content': user_message}],
                    thinking=thinking)
        return self._retry(fn)

    def think_batched(self, system_message: str, user_message: str, thinking: bool = False,
                      max_wait_seconds: float = 3600) -> str | None:
        """`think` for work nobody is waiting on, through the provider's batch API where it has
        one (half price, an answer within minutes to hours). Blocks until the batch ends or
        `max_wait_seconds` pass. None means there is no answer from a batch — no batch API, the
        wait ran out, the request failed there — and the caller makes the ordinary call
        instead. Not retried here: a retry would submit, and pay for, a second batch."""
        return self.create_batched_completion(
            [{'role': 'system', 'content': system_message}, {'role': 'user', 'content': user_message}],
            thinking=thinking, max_wait_seconds=max_wait_seconds)

    def create_batched_completion(self, messages: list[dict], thinking: bool = False,
                                  max_wait_seconds: float = 3600) -> str | None:
        """The completion through the provider's batch API, or None (see `think_batched`).
        Default: no batch API. Override in backends that have one (ClaudeLLM)."""
        return None

    def think_batched_with_images(self, system_message: str, requests: list[tuple[str, bytes]],
                                  max_wait_seconds: float = 1800) -> list[str | None] | None:
        """`think` with one image per request, for a pile of work nobody is waiting on (the photo
        backfill), through the provider's batch API where it has one: half price, the answers
        within minutes to hours. `requests` are (user_message, image bytes) pairs under one shared
        system message. Returns one reply per request in order, None for a request the batch did
        not answer — or None as a whole when there is no batch API or the job gave nothing, and
        the caller makes the ordinary calls instead. Not retried: a retry would pay twice."""
        return self.create_batched_image_completions(system_message, requests, max_wait_seconds=max_wait_seconds)

    def create_batched_image_completions(self, system_message: str, requests: list[tuple[str, bytes]],
                                         max_wait_seconds: float = 1800) -> list[str | None] | None:
        """The image completions through the provider's batch API, or None (see
        `think_batched_with_images`). Default: no batch API. Override in backends that have one."""
        return None

    def supports_video(self) -> bool:
        """True if this LLM accepts a video clip as input. Override in backends that do (Gemini)."""
        return False

    def max_video_bytes(self) -> int:
        """Largest clip think_with_video() can take; 0 when supports_video() is False."""
        return 0

    def think_with_video(self, system_message: str, user_message: str, video_bytes: bytes,
                         mime_type: str, fps: float | None = None) -> str:
        """Sends one video clip plus a prompt, blocks until done, and returns the response.
        Only meaningful when supports_video() is True."""
        raise NotImplementedError(f"{self.name()} cannot take video input")

    def think_with_messages(self, messages: list[dict], thinking: bool = False, volatile_tail: int = 0,
                            follow_up: bool = False) -> str:
        """
        Sends a request to the LLM with a full message history, blocks until done, and returns the response.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            thinking: how much to reason (see `Thinking`); a bool is read as FULL / NORMAL
            volatile_tail: how many trailing messages no later request will send as they are
                (an instruction folded onto the history for this call only). An LLM with a
                prefix cache keeps its history breakpoint off them — a breakpoint on bytes
                nobody sends again is a write that never gets its read.
            follow_up: this call is a later round of one request — it ends in a command result
                the next round of the same request repeats verbatim, seconds from now, and no
                request after that one does. An LLM with a prefix cache caches the whole tail
                for the short tier instead of the long one; volatile_tail is moot then.
        """
        return self._retry(lambda: self.create_completion(messages, thinking=thinking, volatile_tail=volatile_tail,
                                                          follow_up=follow_up))

    @abstractmethod
    def create_streaming_completion(self, messages: list[dict], thinking: bool = False,
                                    volatile_tail: int = 0, follow_up: bool = False) -> Generator[str, None, None]:
        """
        Create a streaming completion from the llm

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            thinking: how much to reason (see `Thinking`); a bool is read as FULL / NORMAL
            volatile_tail: see think_with_messages; only an LLM with a prefix cache acts on it
            follow_up: see think_with_messages; only an LLM with a prefix cache acts on it

        Yields:
            String chunks of the response
        """
        pass

    @abstractmethod
    def create_completion(self, messages: list[dict], thinking: bool = False, volatile_tail: int = 0,
                          follow_up: bool = False) -> str:
        """
        Create a non-streaming completion from the llm

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            thinking: how much to reason (see `Thinking`); a bool is read as FULL / NORMAL
            volatile_tail: see think_with_messages; only an LLM with a prefix cache acts on it
            follow_up: see think_with_messages; only an LLM with a prefix cache acts on it

        Returns:
            String containing the complete response
        """
        pass

    @abstractmethod
    def create_completion_with_images(self, messages: list[dict], images: list, background: bool = False) -> str:
        """
        Create a non-streaming completion with images using OpenAI's API

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            images: List of images in one of these formats:
                - Base64 encoded strings (without data URI prefix)
                - Raw bytes
                - File paths (will be converted to base64)
            background: see think(); only a single-slot backend acts on it

        Returns:
            String containing the complete response
        """
        pass

    def is_local(self) -> bool:
        """True for a backend that serves one request at a time on our own hardware (Ollama):
        bulk callers then run one or two workers instead of a wide fan-out."""
        return False

    def unavailable(self) -> str | None:
        """Why the backend is refusing every request right now (see LLMUnavailable), or None
        when it takes requests. Default: a backend that never tracks this."""
        return None

    def out_of_credits(self) -> bool:
        """True when `unavailable()` is the account's prepaid credits being gone — the one case
        that needs the user, not time, to resolve."""
        return False

    def probe(self) -> bool:
        """Make a request small enough to cost nothing, to learn whether a backend that was
        refusing requests takes them again; True clears `unavailable()`. Default: always True."""
        return True

    def background_wait_seconds(self) -> float:
        """How long the calling thread's last `background=True` request queued before the
        backend took it — 0 for a backend that serves background requests immediately."""
        return 0.0

    @abstractmethod
    def initialize(self):
        """Initialize the llm client"""
        pass

    @abstractmethod
    def name(self) -> str:
        """Return the name of this llm/AI"""
        pass

    def provider(self) -> str:
        """Who runs the server behind this LLM, as the user hears it in an error notice
        ("The Claude server returned an error: …"). name() unless the name carries a model."""
        return self.name()

    def display_model(self) -> str:
        """The model as the user hears it in a notice ("Fable", "Opus"). name() by default."""
        return self.name()

    def api_key_name(self) -> str | None:
        """The config key (api_keys.…) this backend reads, or None for a keyless backend
        (Ollama). Known from the class alone — an integration that runs on an LLM tier
        declares this key as the one it needs."""
        return None

    def missing_api_key(self) -> str | None:
        """The config key (api_keys.…) this backend needs but has not got, or None when it can
        run. A caller that would rather skip its job than fail on initialize() asks this first."""
        return None

    def refusal_fallback(self) -> "AbstractLLM | None":
        """Another LLM to answer a request this one's safety layer refused, or None when
        the refusal should reach the user as-is. Used once per request — the fallback's
        own refusal is final."""
        return None
