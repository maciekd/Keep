"""Runs a local model on any server that speaks the OpenAI chat API — oMLX, llama.cpp's server, ds4-server through its subclass — with the prefill bar, the cache report and the single-slot gate every local backend gets.

The server is the `openai_compatible:` section of the config: `base_url` (default
http://127.0.0.1:8000/v1, where oMLX and ds4-server listen), `model` (an id or alias of
`GET /v1/models`; unset, the first one the server lists), an optional `api_key` (the SDK refuses
to start without one, so a placeholder goes out when the server takes none), `reasoning_effort`
and `background_idle_seconds`. The root `local_backend: openai` routes every tier here.
`initialize` lists the server's models once: a configured model the server does not list is
logged with the ones it has, and a `max_model_len` the server reports (oMLX does) is logged too.

The model reasons server-side when the server lets it: the stream's `reasoning_content` deltas
are yielded inside <thinking> tags so the command processor swallows and times them like a
prompted thinking block, and `reasoning_effort` goes with every request, from the request's
`Thinking`: FULL (the memory curator) gets "high"; NORMAL (a conversation turn) the config key,
default "low"; MORE (the autonomous check) one step above that key, "medium" at least. "none"
turns server-side reasoning off, and the prompt's own <thinking> instruction applies instead.

No request declares a tool: the model writes every call as text in the prompt's syntax (the
tier's `command_format:`, Keep's own by default) and the command processor reads it from the
content. A server that hands back parsed `tool_calls` all the same gets them written back as
call text in the active format, at the end of the reply.

The "Prefilling…" bar: every request goes out with `return_progress: true` (llama.cpp's
extension, oMLX's too; `openai_compatible.return_progress: false` for a server that rejects
the key) and the server streams `prompt_progress` chunks while it prefills — the prompt's
total, cached and processed tokens and the elapsed time — which the class's one
`PrefillTracker` turns into the bar, the rate beside it and the early cache-break report, the
way ds4 and Ollama get them from their logs. A server that streams no progress leaves the bar
on the tracker's time estimate. The stream also asks for `stream_options.include_usage`, and
the closing usage chunk is the metrics line of the request.

Background work (photo indexing, `background=True`): a local server serves one request at a
time, so an image call goes through the single-slot gate every instance of the class shares
(`BackgroundSlot`, Ollama's): it waits until nothing interactive is in flight and the server
has been idle `background_idle_seconds`, streams over a raw connection of its own, and a user
turn arriving mid-image shuts that socket and takes the slot; the evicted caller gets
LLMPreempted and resends when the slot is idle again. The pre-heat (`pre_heat_cache`, fired by
`PromptManager` whenever the byte prefix of the next request changes) is a streamed one-token
request carrying the new prefix, so a server with a prefix cache has it warm when the next
user turn arrives; one at a time, the latest pending prompt only.

The subclass hooks: `_prepare_messages` (ds4 replays its history), `_content` (ds4 withholds a
DSML block), `_release_withheld`, `_remember`, `_configure_prefill` (ds4 points the tracker at
its log). Everything else here is the OpenAI API as the local servers speak it.
"""
import json
import threading
import time
from types import SimpleNamespace
from typing import Generator
from urllib.parse import urlparse

import httpx
from openai import OpenAI

from .AbstractLLM import LLMPreempted, Thinking
from .BackgroundSlot import BackgroundSlot
from .ChatGPTLLM import ChatGPTLLM
from .PrefillTracker import PrefillTracker
from assistant.commands.formats import QwenFormat as qwen
from assistant.commands.formats.CommandFormats import CommandFormats
from assistant.commands.formats.KeepFormat import KeepFormat
from common.ConfigManager import config
from common.logging import error, info, trace, fine, note


class _Content:
    """The reply text as the server streams it: what a generic server sends is the reply, so
    every delta is visible at once and nothing is held back. ds4 swaps in a DSML-aware one."""

    withheld = ""

    def take(self, text: str) -> str:
        """The part of a content delta to yield now."""
        return text

    def flush(self) -> str:
        """Text held back for a split marker, to yield once the stream is over."""
        return ""

    def split(self, text: str) -> tuple[str, str]:
        """A non-streamed reply: the text to show, and the block text to withhold from it."""
        return text, ""


class OpenAICompatibleLLM(ChatGPTLLM):
    """Any local OpenAI-compatible server as an LLM backend: oMLX, llama.cpp, or ds4-server
    through `Ds4LLM`."""

    SECTION = 'openai_compatible'      # the config section with base_url, model, reasoning_effort …
    NAME = "OpenAI-compatible"         # what the user hears in a notice
    LABEL = "openai-compatible"        # what the log lines carry
    DEFAULT_MODEL: str | None = None   # unset: the first model the server lists
    DEFAULT_BASE_URL = 'http://127.0.0.1:8000/v1'
    RETURN_PROGRESS = True             # ask for `prompt_progress` chunks (a subclass with a log needs none)
    # Measured on an M-series Mac with a 27B model at 4 bits; each reply refines it.
    _prefill = PrefillTracker("openai-compatible", None, seed_tps=800.0)
    DEFAULT_REASONING_EFFORT = 'low'
    THINKING_REASONING_EFFORT = 'high'   # Thinking.FULL
    MORE_REASONING_EFFORT = 'medium'     # the floor of Thinking.MORE
    # OpenAI's spellings, lowest first; "minimal" is the smallest non-zero level, "none" alone disables reasoning.
    REASONING_EFFORTS = ('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max')
    _slot = BackgroundSlot("openai-compatible")
    # The pre-heat guard, shared by the tiers' instances like the slot (module docstring); a
    # subclass declares its own set, since it serves another server.
    _pre_heat_lock = threading.Lock()
    _pre_heat_in_flight = False
    _pre_heat_pending: list[dict] | None = None
    PRE_HEAT_MAX_TOKENS = 1
    PRE_HEAT_READ_TIMEOUT_S = 600.0
    MODELS_READ_TIMEOUT_S = 10.0
    _logged_configs: set[tuple[str | None, str]] = set()
    _models_checked: set[tuple[str, str]] = set()
    _discovered_models: dict[str, str] = {}      # base_url → the first listed model, for every instance with none set
    _discovery_failed: set[str] = set()          # the base_urls whose empty list was already an error line
    CONNECT_TIMEOUT_S = 5.0

    def __init__(self, model: str | None = None, base_url: str | None = None):
        """`model` is the tier's `model:` when it has one, else the section's, else the default."""
        cls = type(self)
        super().__init__(model=model or config.get(f'{cls.SECTION}.model', cls.DEFAULT_MODEL or '') or None,
                         api_key_name=cls.SECTION,
                         base_url=base_url or config.get(f'{cls.SECTION}.base_url', cls.DEFAULT_BASE_URL))
        if (self.model, self.base_url) not in cls._logged_configs:
            cls._logged_configs.add((self.model, self.base_url))
            info(f"{cls.__name__} '{self.model or '(the server’s first model)'}' using {self.base_url}, "
                 f"reasoning effort {self.reasoning_effort()}")
        self._configure_prefill()

    def _configure_prefill(self):
        """Where the prefill bar's numbers come from: the server's own progress chunks here, so
        nothing to set up. A subclass whose server says nothing on the wire points the
        tracker at the server's log instead."""

    def initialize(self):
        """No read timeout: a non-streamed reply says nothing until it is done, and the memory
        curation (a whole bank rewritten at effort high) runs past the SDK's 600 s — which then
        resent it, a fresh generation each time, to a server that serves one at a time."""
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                             timeout=httpx.Timeout(None, connect=self.CONNECT_TIMEOUT_S))
        trace(f"{self.LABEL} client: no read timeout, {self.CONNECT_TIMEOUT_S:.0f} s to connect")
        self._check_model()

    def _check_model(self):
        """`GET /v1/models`, once per server and model: the configured model is checked against
        the list — a miss is logged with the ids the server has, not refused, since a server
        may take a name it does not list — and an unset model becomes the first listed one,
        for this instance and every later one of the same server (the dialog builds one per
        tier and request, and each must name the model). A list that could not be read pins
        a configured model as checked, but leaves an unset one to the next instance's try.
        The server also tells the context length there when it knows it (`max_model_len`)."""
        cls = type(self)
        if self.model is None and self.base_url in cls._discovered_models:
            self.model = cls._discovered_models[self.base_url]
            trace(f"{self.LABEL}: no model configured — '{self.model}', the one {self.base_url} listed first, stands")
            return
        if (self.model, self.base_url) in cls._models_checked:
            return
        try:
            client = self.client.with_options(timeout=httpx.Timeout(self.MODELS_READ_TIMEOUT_S, connect=self.CONNECT_TIMEOUT_S))
            listed = [m for m in client.models.list() if isinstance(getattr(m, 'id', None), str)]
        except Exception as e:
            trace(f"{self.LABEL}: the server's model list could not be read ({e}) — the configured model stands unchecked")
            listed = []
        if not listed:
            if self.model is None:
                log = trace if self.base_url in cls._discovery_failed else error
                cls._discovery_failed.add(self.base_url)
                log(f"{self.LABEL}: no model configured ({self.SECTION}.model) and {self.base_url} lists none — "
                    f"the requests go out without one, which a server serving one model may take")
            else:
                cls._models_checked.add((self.model, self.base_url))
                trace(f"{self.LABEL}: the server lists no models — '{self.model}' stands unchecked")
            return
        ids = [m.id for m in listed]
        if self.model is None:
            self.model = cls._discovered_models[self.base_url] = ids[0]
            info(f"{self.LABEL}: no model configured — using the first the server lists, '{self.model}'"
                 f"{f' (of {len(ids)})' if len(ids) > 1 else ''}")
        cls._models_checked.add((self.model, self.base_url))
        if self.model not in ids:
            note(f"{self.LABEL}: '{self.model}' is not among the models {self.base_url} lists ({', '.join(ids)}) — "
                 f"sending it anyway; set {self.SECTION}.model to one of them if the requests fail")
        mine = next((m for m in listed if m.id == self.model), None)
        length = getattr(mine, 'max_model_len', None) if mine is not None else None
        if isinstance(length, int):
            info(f"{self.LABEL}: '{self.model}' takes {length} tokens of context")

    def _note_stop(self, finish_reason: str | None):
        """Keep the reply's stop reason where `last_stop_reason` reads it, in Anthropic's spelling
        for a reply cut at the output cap (`length` → `max_tokens`), so the curator's check sees it."""
        self._last_stop_reason = 'max_tokens' if finish_reason == 'length' else finish_reason
        if self._last_stop_reason == 'max_tokens':
            info(f"{self.LABEL}: the reply was cut off at the output limit (finish_reason=length)")

    @property
    def api_key(self) -> str:
        """`<section>.api_key` when the server wants one; the OpenAI client refuses to start
        without a key, so a placeholder goes out to a server that ignores it."""
        return config.get(f'{self.SECTION}.api_key', '') or 'local'

    def api_key_name(self) -> None:
        return None

    def missing_api_key(self) -> None:
        return None

    def name(self) -> str:
        return self.NAME

    def display_model(self) -> str:
        return self.model or "the local model"

    def is_local(self) -> bool:
        return True

    def default_command_format(self):
        """Qwen's <tool_call> syntax for a Qwen model (what the family was trained on, and what
        `OllamaLLM` writes for the same models), Keep's own call text for any other; the tier's
        `command_format:` key wins over both. Read from the model name, so a model left to
        `initialize` to discover is Keep's syntax until the server has been reached once."""
        return qwen.QwenFormat if 'qwen' in (self.model or '').lower() else KeepFormat

    def prefers_prepended_user_query(self) -> bool:
        """A local server's prefix cache matches from byte 0: a query prepended to the system
        prompt changes the first bytes every turn and costs the whole cache."""
        return False

    def prefers_byte_prefix_tail(self) -> bool:
        """A prefix cache reuses the previous request only as far as the new one repeats it
        token for token; a trailing directive folded onto the last message is dropped when
        the next round renders that message as history, so the prefix would diverge before
        the previous prompt's end every round."""
        return True

    @classmethod
    def reasoning_effort(cls) -> str:
        """The effort of an ordinary request (thinking=False): `<section>.reasoning_effort`, or
        the default; a spelling the API does not know falls back to the default and is logged."""
        effort = str(config.get(f'{cls.SECTION}.reasoning_effort', cls.DEFAULT_REASONING_EFFORT)).lower()
        if effort not in cls.REASONING_EFFORTS:
            info(f"{cls.SECTION}.reasoning_effort '{effort}' is not one of {', '.join(cls.REASONING_EFFORTS)} — "
                 f"using {cls.DEFAULT_REASONING_EFFORT}")
            return cls.DEFAULT_REASONING_EFFORT
        return effort

    @classmethod
    def returns_progress(cls) -> bool:
        """Whether requests ask the server for `prompt_progress` chunks (`<section>.return_progress`)."""
        return bool(config.get(f'{cls.SECTION}.return_progress', cls.RETURN_PROGRESS))

    def supports_native_thinking(self) -> bool:
        return True

    def handles_thinking_internally(self) -> bool:
        """The model reasons on its own unless the effort is "none"."""
        return self.reasoning_effort() != 'none'

    def _effort_for(self, thinking: Thinking | bool) -> str:
        """The server's effort for a request: FULL is high, NORMAL the config key, MORE one step
        above the key and never below medium (max stays max)."""
        thinking = Thinking.of(thinking)
        if thinking.full:
            return self.THINKING_REASONING_EFFORT
        effort = self.reasoning_effort()
        if thinking is Thinking.MORE:
            levels = self.REASONING_EFFORTS
            raised = levels[min(levels.index(effort) + 1, len(levels) - 1)]
            return max(raised, self.MORE_REASONING_EFFORT, key=levels.index)
        return effort

    @classmethod
    def background_idle_seconds(cls) -> float:
        return float(config.get(f'{cls.SECTION}.background_idle_seconds', BackgroundSlot.DEFAULT_IDLE_S))

    def background_wait_seconds(self) -> float:
        return self._slot.wait_seconds()

    # --- the pre-heat -------------------------------------------------------------------------

    def pre_heat_cache(self, messages: list[dict]):
        """Prefill the prompt's new prefix in the background so the next turn finds it cached
        (module docstring). At most one in flight; a prompt arriving meanwhile replaces the
        pending one and goes out when the running pre-heat ends."""
        cls = type(self)
        with cls._pre_heat_lock:
            if cls._pre_heat_in_flight:
                cls._pre_heat_pending = messages
                trace(f"{self.LABEL} pre-heat deferred: one is in flight — this prompt goes out when it ends")
                return
            cls._pre_heat_in_flight = True
        self._fire_pre_heat(messages)

    def _fire_pre_heat(self, messages: list[dict]):
        threading.Thread(target=self._run_pre_heat, args=(messages,), name=f"{type(self).__name__}PreHeat",
                         daemon=True).start()

    def _run_pre_heat(self, messages: list[dict]):
        try:
            self._pre_heat(messages)
        except Exception as e:
            trace(f"{self.LABEL} pre-heat failed (non-critical): {e}")
        finally:
            self._finish_pre_heat()

    def _finish_pre_heat(self):
        """Release the guard, or hand it straight to the prompt deferred meanwhile."""
        cls = type(self)
        with cls._pre_heat_lock:
            pending, cls._pre_heat_pending = cls._pre_heat_pending, None
            if pending is None:
                cls._pre_heat_in_flight = False
        if pending is not None:
            info(f"{self.LABEL} pre-heat: sending the prompt deferred meanwhile")
            self._fire_pre_heat(pending)

    def _pre_heat(self, messages: list[dict]):
        """The request: the prepared messages as a real turn sends them, the chat tier's effort,
        no tool, streamed, one output token. The reply is dropped — it is no turn, so neither
        the subclass's store nor the cache report sees it. Gated as interactive so a
        photo-indexing request waits, and shown as background on the prefill bar. Unlike a real
        turn it has a read timeout (PRE_HEAT_READ_TIMEOUT_S, Ollama's): a half-open socket would
        otherwise park every later pre-heat and the interactive slot for the life of the process."""
        if self.client is None:
            self.initialize()
        client = self.client.with_options(
            timeout=httpx.Timeout(self.PRE_HEAT_READ_TIMEOUT_S, connect=self.CONNECT_TIMEOUT_S))
        if not any(m.get('role') == 'user' for m in messages):
            # A prompt with no user turn yet: the stub stands after the shared system prefix,
            # which is what the pre-heat is for; the real query displaces it at the tail.
            messages = messages + [{'role': 'user', 'content': ' '}]
        kwargs = self._request_kwargs(messages, Thinking.NORMAL, stream=True)
        kwargs['max_tokens'] = self.PRE_HEAT_MAX_TOKENS
        info(f"{self.LABEL} pre-heat: prefilling the prompt's new prefix ({len(messages)} messages, "
             f"effort {kwargs['extra_body']['reasoning_effort']})...")
        started = time.monotonic()
        self._slot.interactive_begin()
        token = self._prefill.begin(interactive=False)
        try:
            usage = None
            for chunk in client.chat.completions.create(**kwargs):
                if getattr(chunk, 'usage', None) is not None:
                    usage = chunk.usage
                self._take_progress(token, chunk)
            # Read before `end` drops the request from the tracker.
            seconds = self._prefill.prefill_seconds(token) if usage is not None else None
        finally:
            self._prefill.end(token)
            self._slot.interactive_end()
        elapsed = time.monotonic() - started
        if usage is None:
            info(f"{self.LABEL} pre-heat complete in {elapsed:.1f} s (the server reported no usage)")
            return
        cached = self._cached_tokens(usage)
        prefilled = usage.prompt_tokens - (cached or 0)
        info(f"{self.LABEL} pre-heat complete: {usage.prompt_tokens} tokens, {cached if cached is not None else '?'} cached, "
             f"{prefilled} prefilled in {elapsed:.1f} s")
        if seconds:
            self._prefill.observe_rate(prefilled, seconds)

    # --- the request ---------------------------------------------------------------------------

    def _prepare_messages(self, messages: list[dict]) -> list[dict]:
        """The converted messages as this server wants them; the hook a subclass rewrites the
        history in (ds4's replay)."""
        return messages

    def _request_kwargs(self, messages: list[dict], thinking: Thinking | bool, stream: bool) -> dict:
        messages = self._prepare_messages(self._convert_anthropic_content(messages))
        # No `tools`: see the module docstring — the model writes its calls as text in the prompt's syntax.
        extra = {'reasoning_effort': self._effort_for(thinking)}
        kwargs = dict(model=self.model, messages=messages, stream=stream, extra_body=extra)
        if stream:
            kwargs['stream_options'] = {'include_usage': True}
            if self.returns_progress():
                extra['return_progress'] = True
        return kwargs

    @staticmethod
    def _cached_tokens(usage) -> int | None:
        details = getattr(usage, 'prompt_tokens_details', None) if usage is not None else None
        return getattr(details, 'cached_tokens', None) if details is not None else None

    def _log_usage(self, usage, started: float, reasoning_chars: int = 0) -> int | None:
        """One line per request: prompt tokens with the server's cache hits, output tokens, time.
        Returns the tokens the server prefilled (prompt minus cached), None when it said nothing."""
        if usage is None:
            trace(f"{self.LABEL}: response carried no usage")
            return None
        cached = self._cached_tokens(usage)
        cached_text = f", {cached} cached" if cached is not None else ""
        thought = f", {reasoning_chars} reasoning chars" if reasoning_chars else ""
        info(f"{self.LABEL} metrics: ↑ {usage.prompt_tokens} tokens{cached_text} | ↓ {usage.completion_tokens} tokens{thought} "
             f"| {time.monotonic() - started:.2f}s")
        return usage.prompt_tokens - (cached or 0)

    def cache_scope(self) -> str | None:
        return f"{self.LABEL} {self.base_url}"

    def _take_progress(self, token: int | None, chunk) -> bool:
        """A streamed chunk's `prompt_progress`, when it carries one, handed to the prefill
        tracker (the SDK keeps the field the schema lacks as an extra; a raw SSE chunk is a
        dict). True when the chunk was a progress chunk."""
        progress = chunk.get('prompt_progress') if isinstance(chunk, dict) else getattr(chunk, 'prompt_progress', None)
        if not progress:
            return False
        read = progress.get if isinstance(progress, dict) else lambda k, d=None: getattr(progress, k, d)
        total, processed = read('total'), read('processed')
        if not isinstance(total, int) or not isinstance(processed, int):
            trace(f"{self.LABEL}: a prompt_progress chunk without total/processed ({progress!r}) — ignored")
            return True
        cache, time_ms = read('cache', 0), read('time_ms')
        self._prefill.report_progress(token, total, int(cache or 0), processed,
                                      float(time_ms) if isinstance(time_ms, (int, float)) else None)
        return True

    def _cache_report(self, messages: list[dict], effort: str):
        """One request's cache-break report, made once: early, from the workload the server
        reports as the prefill starts (its first progress chunk, or a log line) — the Home
        tab's row then lands while the bar is still drawn — or, when it reports none, at the
        end from the reply's usage. (early, late): `early` is the tracker's `on_work` callback,
        `late` takes the usage and reports only when `early` never ran. The two can race on a
        poll that finishes as the stream ends, hence the lock."""
        lock, made_from = threading.Lock(), []

        def once(cached: int, prompt_tokens: int, source: str):
            with lock:
                if made_from:
                    trace(f"{self.LABEL} cache: the {source} says {cached}/{prompt_tokens} cached — "
                          f"already reported from the {made_from[0]}")
                    return
                made_from.append(source)
            self.record_cache_hit(cached, prompt_tokens)
            self._report_cache(messages, cached, prompt_tokens, effort)

        def early(cached: int, remaining: int):
            once(cached, cached + remaining, "workload line")

        def late(usage):
            cached = self._cached_tokens(usage)
            if cached is None:
                trace(f"{self.LABEL} cache: the reply carries no cached-token count — no report")
                return
            once(cached, usage.prompt_tokens, "usage")

        return early, late

    def _report_cache(self, messages: list[dict], cached: int, prompt_tokens: int, effort: str):
        """The cached tokens against the prompt, for the cache-break report. A server renders
        past turns by the reasoning effort, so an effort other than the previous request's is
        the note the report falls back on when the prompt text is unchanged."""
        previous, self._last_effort = getattr(self, '_last_effort', None), effort
        note_text = None
        if previous is not None and previous != effort:
            note_text = f"the reasoning effort changed from {previous} to {effort}, which re-renders the past turns"
        self.report_cache_usage(messages, cached, prompt_tokens, note=note_text)

    # --- the reply -----------------------------------------------------------------------------

    def _content(self) -> _Content:
        """How a request's content deltas are read (see `_Content`)."""
        return _Content()

    def _rendered_calls(self, calls) -> list[tuple[str, object]]:
        """Parsed tool calls (the merged stream deltas, or a message's `tool_calls`) each
        written back as call text in the active command format, so the parser reads it —
        paired with the call; an entry without a name is dropped."""
        fmt = CommandFormats.active()
        rendered = []
        for call in calls:
            function = getattr(call, 'function', None)
            name = getattr(function, 'name', None)
            if not name:
                trace(f"{self.LABEL} tool call without a function name, dropped: {call!r}")
                continue
            arguments = getattr(function, 'arguments', None) or "{}"
            trace(f"{self.LABEL} parsed a {name} call — written back as {fmt.name} call text")
            rendered.append((fmt.render_tool_call(name, arguments), call))
        return rendered

    @staticmethod
    def _tail(rendered: list[tuple[str, object]]) -> str:
        return "\n".join(text for text, _ in rendered)

    def _closing(self, reply: str, withheld: str, rendered: list[tuple[str, object]]) -> tuple[str, str]:
        """What follows the reply text — the parsed calls as call text on a line of their own, or
        the withheld block released as written — and the content as the model wrote it (before a
        parsed call: the text the server kept, trailing whitespace trimmed)."""
        if rendered:
            tail = self._tail(rendered)
            return ("\n" + tail if reply and not reply.endswith("\n") else tail), reply.rstrip()
        released = self._release_withheld(withheld)
        return released, reply + released

    def _release_withheld(self, withheld: str) -> str:
        """Text a subclass's `_Content` withheld, once the reply is complete and no parsed call
        was rendered in its place. Nothing is withheld here."""
        return withheld

    def _remember(self, reply: str, reasoning: str, verbatim: str):
        """A finished reply, for a subclass that keeps what the model wrote (ds4's replay store)."""

    def create_streaming_completion(self, messages: list[dict], thinking: Thinking | bool = Thinking.NORMAL,
                                    volatile_tail: int = 0, follow_up: bool = False) -> Generator[str, None, None]:
        """Reasoning deltas (`reasoning_content`) are yielded wrapped in <thinking> tags, the reply
        text after them through the class's `_Content`; the `tool_calls` deltas (merged by index)
        are yielded as call text in the active format at the end. The stream's progress chunks
        feed the prefill bar and its closing usage chunk is logged."""
        self._ensure_initialized()
        started = time.monotonic()
        self._slot.interactive_begin()
        early, late = self._cache_report(messages, self._effort_for(thinking))
        token = self._prefill.begin(on_work=early)
        try:
            yield from self._stream(messages, thinking, started, token, late)
        finally:
            self._prefill.end(token)
            self._slot.interactive_end()

    def _stream(self, messages: list[dict], thinking: Thinking | bool, started: float, token: int | None,
                report_cache) -> Generator[str, None, None]:
        self._last_stop_reason = None
        response = self.client.chat.completions.create(**self._request_kwargs(messages, thinking, stream=True))
        in_thinking, reasoning, usage = False, "", None
        reply, content = "", self._content()
        calls: dict[int, _ToolCall] = {}
        prefilling = True
        for chunk in response:
            if getattr(chunk, 'usage', None) is not None:
                usage = chunk.usage
            self._take_progress(token, chunk)
            if not chunk.choices:
                continue
            if chunk.choices[0].finish_reason:
                self._note_stop(chunk.choices[0].finish_reason)
            delta = chunk.choices[0].delta
            if prefilling and (delta.content or getattr(delta, 'reasoning_content', None)
                               or getattr(delta, 'tool_calls', None)):
                prefilling = False
                self._prefill.first_token(token)
            thought = getattr(delta, 'reasoning_content', None)
            if thought:
                if not in_thinking:
                    in_thinking = True
                    yield "<thinking>"
                reasoning += thought
                yield thought
            if delta.content:
                if in_thinking:
                    in_thinking = False
                    yield "</thinking>"
                visible = content.take(delta.content)
                if visible:
                    reply += visible
                    yield visible
            for entry in getattr(delta, 'tool_calls', None) or []:
                index = getattr(entry, 'index', None) or 0
                calls.setdefault(index, _ToolCall()).merge(entry)
        if in_thinking:
            trace(f"{self.LABEL} stream ended inside the thinking block — closing it")
            yield "</thinking>"
        held = content.flush()
        if held:
            reply += held
            yield held
        if content.withheld:
            trace(f"{self.LABEL} withheld {len(content.withheld)} chars of the reply ({len(calls)} tool call(s) parsed)")
        rendered = self._rendered_calls([calls[i] for i in sorted(calls)])
        tail, verbatim = self._closing(reply, content.withheld, rendered)
        if tail:
            yield tail
        self._remember(reply + tail, reasoning, verbatim)
        prefilled = self._log_usage(usage, started, len(reasoning))
        report_cache(usage)
        if prefilled is not None:
            seconds = self._prefill.prefill_seconds(token)
            if seconds:
                self._prefill.observe_rate(prefilled, seconds)

    def create_completion(self, messages: list[dict], thinking: Thinking | bool = Thinking.NORMAL, volatile_tail: int = 0,
                          follow_up: bool = False) -> str:
        """The reply text, then every parsed tool call written back as call text; the model's
        reasoning is logged, not returned."""
        self._ensure_initialized()
        started = time.monotonic()
        self._slot.interactive_begin()
        early, late = self._cache_report(messages, self._effort_for(thinking))
        token = self._prefill.begin(on_work=early)
        self._last_stop_reason = None
        try:
            response = self.client.chat.completions.create(**self._request_kwargs(messages, thinking, stream=False))
        finally:
            self._prefill.end(token)
            self._slot.interactive_end()
        self._note_stop(response.choices[0].finish_reason)
        message = response.choices[0].message
        reasoning = getattr(message, 'reasoning_content', None) or ""
        if reasoning:
            fine(f"{self.LABEL} thought:\n{reasoning.strip()}")
        self._log_usage(getattr(response, 'usage', None), started, len(reasoning))
        late(getattr(response, 'usage', None))
        calls = getattr(message, 'tool_calls', None) or []
        content, withheld = self._content().split(message.content or "")
        rendered = self._rendered_calls(calls)
        tail, verbatim = self._closing(content, withheld, rendered)
        self._remember(content + tail, reasoning, verbatim)
        return content + tail

    # --- images --------------------------------------------------------------------------------

    def create_completion_with_images(self, messages: list[dict], images: list, background: bool = False) -> str:
        """A background image request (photo indexing) goes through the slot gate over a raw
        connection the gate can shut; an interactive one is the plain call, evicting any
        background request first."""
        if background:
            return self._background_completion(self._image_messages(messages, images))
        self._slot.interactive_begin()
        try:
            return super().create_completion_with_images(messages, images)
        finally:
            self._slot.interactive_end()

    def _background_completion(self, messages: list[dict]) -> str:
        """The abortable image request: waits for the idle slot, streams the answer over its
        own connection (SSE, the OpenAI shape: `data: {chunk}` lines, `data: [DONE]` last, a
        bare comment as the keepalive while the server prefills) and raises LLMPreempted when
        a user request evicts it. No tool is declared: an image request matches no cached
        prefix anyway, and its reply is text."""
        request = {'model': self.model, 'messages': messages, 'stream': True,
                   'stream_options': {'include_usage': True}, 'reasoning_effort': self.reasoning_effort()}
        if self.returns_progress():
            request['return_progress'] = True
        body = json.dumps(request)
        path = urlparse(self.base_url).path.rstrip('/') + '/chat/completions'
        slot = self._slot
        slot.acquire(self.background_idle_seconds())
        started = time.monotonic()
        early, late = self._cache_report(messages, self.reasoning_effort())
        token = self._prefill.begin(interactive=False, on_work=early)
        chunks: list[str] = []
        reasoning_chars, usage = 0, None
        try:
            conn = slot.open_connection(self.base_url)
            slot.register_connection(conn)
            headers = {"Content-Type": "application/json"}
            if config.get(f'{self.SECTION}.api_key', ''):
                headers["Authorization"] = f"Bearer {self.api_key}"
            conn.request("POST", path, body=body, headers=headers)
            resp = conn.getresponse()
            if resp.status != 200:
                raise RuntimeError(f"{self.LABEL} {path} returned {resp.status}: {resp.read(500)!r}")
            for raw in iter(resp.readline, b""):
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue    # the SSE keepalive comment, or a blank line
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                data = json.loads(payload)
                if data.get('usage'):
                    usage = _Usage(data['usage'])
                self._take_progress(token, data)
                for choice in data.get('choices') or []:
                    delta = choice.get('delta') or {}
                    if delta.get('reasoning_content'):
                        self._prefill.first_token(token)
                        reasoning_chars += len(delta['reasoning_content'])
                    if delta.get('content'):
                        self._prefill.first_token(token)
                        chunks.append(delta['content'])
            else:
                raise RuntimeError(f"{self.LABEL} closed the stream before the response was done")
            conn.close()
        except LLMPreempted:
            raise
        except Exception:
            if preempted := slot.preempted(len(chunks)):
                raise preempted from None
            raise
        finally:
            self._prefill.end(token)
            slot.release()
        self._log_usage(usage, started, reasoning_chars)
        late(usage)
        return "".join(chunks)


class _Usage:
    """The usage object of a raw SSE chunk in the SDK's attribute shape, for `_log_usage`."""

    def __init__(self, data: dict):
        self.prompt_tokens = data.get('prompt_tokens', 0)
        self.completion_tokens = data.get('completion_tokens', 0)
        details = data.get('prompt_tokens_details') or {}
        self.prompt_tokens_details = SimpleNamespace(cached_tokens=details.get('cached_tokens'))


class _ToolCall:
    """One tool call assembled from stream deltas sharing an index: the name and id are set
    once, the argument text is concatenated."""

    def __init__(self):
        self.id = None
        self.function = _Function()

    def merge(self, delta):
        self.id = self.id or getattr(delta, 'id', None)
        function = getattr(delta, 'function', None)
        if function is not None:
            self.function.name = self.function.name or getattr(function, 'name', None)
            self.function.arguments += getattr(function, 'arguments', None) or ""

    def __repr__(self):
        return f"_ToolCall(id={self.id!r}, name={self.function.name!r}, arguments={self.function.arguments!r})"


class _Function:
    def __init__(self):
        self.name = None
        self.arguments = ""
