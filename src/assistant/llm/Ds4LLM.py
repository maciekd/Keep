"""Talks to a local ds4-server through its OpenAI-compatible API and runs Keep's commands in Keep's own call text.

ds4 (antirez's DwarfStar engine) serves one large open-weight model — DeepSeek V4.1 Flash by
default — resident on this machine's GPU, at `ds4.base_url` (default http://127.0.0.1:8000/v1);
`model` is the id the server loaded (`/v1/models`). The server takes no API key, so the client
sends a placeholder and this backend is keyless for the setup code (`api_key_name()` is None,
like Ollama).

The model reasons server-side: the stream carries `reasoning_content` deltas, yielded here
inside <thinking> tags so the command processor swallows and times them like a prompted
thinking block. How much it reasons is `reasoning_effort`, from the request's `Thinking`:
FULL (the memory curator) gets "high"; NORMAL (a conversation turn) the `ds4.reasoning_effort`
key, default "low" so the turn answers in seconds; MORE (the autonomous check, which nobody
waits on) one step above that key, "medium" at least — its prompt then shares the chat's
cached prefix only up to the first assistant turn of the history, since the effort changes
how the server renders past turns, and the check prefills the rest itself. "none" turns
server-side reasoning off, and then the prompt's own <thinking> instruction applies instead.

Commands: the default format is Keep's own call text (`KeepFormat`), the best of the formats
on the prompt test suite with V4.1 Flash (measured 2026-10-22: 96.1 / 97.2 % pass at effort
low, 7–8 s a test); the tier's `command_format:` key picks another. No request declares a
tool, so the model writes every call as text in the prompt's syntax and the command processor
reads it from the content. Whatever the format, the model may still write a DSML block
(`DsmlFormat`, the V4.1 spelling); with no tool declared ds4-server leaves it in the content,
and the stream withholds it from the first DSML sentinel on and releases it whole at the end,
as written, for the processor — which reads it under any format, as the DSML rescue of
`RescueFormat` (the slips of 2026-11-07 included), and stores the turn in the prompt's syntax.
Should the server hand back parsed `tool_calls` all the same — it does read a well-formed block
whether or not a tool is declared —, they are written back as call text in the active format
and yielded.

No tool, on purpose (2026-10-22). One
declared tool — Ollama's static `load_toolkit`, which Ollama needs to parse calls at all —
switched ds4 to its tool rendering, which keeps a past call only as a `tool_calls` entry by the
id ds4 assigned; Keep's history holds the call as text, so a command round never extended the
live context and loaded the nearest disk prefix instead. And the model's call head picked among
the declared tools: where its reasoning named a command of a toolkit it had just loaded, it
often wrote `load_toolkit` for that toolkit again. Replaying the day's drifting prompts: 7 of 16
such repeats with the tool, 1 of 16 without; over a fresh dialog 1.9k tokens prefilled per
follow-up request with it, 0.8k without, 243 s of server time against 129 s. With no tool the
server renders past turns without their reasoning and continues its live context from a
thinking checkpoint whenever the request's rendered text extends the visible text the model
wrote — so every assistant row goes out as the model wrote it, whitespace included, which
`Ds4Replay` restores from a side store this class fills after every reply (Keep's history strips
the rows). It also streams a text reply as it is written: with a tool declared and thinking on,
ds4-server held a whole text reply back until generation ended (its second-reasoning guard).

The replay covers one miss only: the next turn of the same dialog. Any other prompt in between
(another tier, the autonomous check, a second dialog) or a server restart evicts the live
context, and that is the server's own fix — start it with a disk KV cache (`scripts/local_ds4_start.sh`
in the repository root does):

    ./ds4-server --kv-disk-dir ~/.ds4/server-kv --kv-disk-space-mb 65536

The server then saves whole prefixes (the first prompt cut back to its last user turn, every
10k aligned tokens of a long prefill, the live state whenever another prompt evicts it, and at
shutdown), and a request loads the longest saved prefix of its prompt and prefills the rest.
Measured on the test host, 2026-10-20, a 16.5k-token prompt: the switch between two prompts
12.3 s → 0.2 s on V4.1 Flash, 31 s → 0.17 s on GLM 5.2. One saved entry is ~111 MiB on Flash and
~517 MiB on GLM at that length, ~253 MiB on Flash at Keep's 40k; the server's default budget is
4 GB. The budget must stay well ahead of the saves: a full cache evicts the entry with no hits,
which is the cold checkpoint a new prompt just stored, and that prompt then prefills from zero on
every request (31 s at 40k; seen 2026-10-21 when the test suite's reasoning runs — a new prompt
each, since the effort changes how the history renders — filled 16 GB in ~60 switches). The
launcher's 64 GB holds ~250. The two fixes are complementary, not alternatives: the replay keeps
the live context through a run of rounds, the disk cache brings a turn after another prompt
back — a new user turn after an unrelated request restored all but 44 tokens of a 19k-token
history (2026-10-22, the cache freshly emptied). The server's eviction knows no age, so a
checkpoint nothing uses again stays until the budget is full and then outlives every fresh one
(`Ds4DiskCacheSweeper` has the numbers): the first client of the process starts the hourly
sweep that removes the entries unused for `ds4.kv_disk_max_age_hours`.

The third piece is the pre-heat (`pre_heat_cache`, fired by `PromptManager` through the launcher's
callback whenever the byte prefix of the next request changes — a prompt rebuild, a section note,
a finished turn): a streamed one-token request carrying the new prefix at the chat tier's effort,
so the server prefills it while it is idle and the next user turn finds it as the live context
and prefills its own tail alone. A user turn arriving meanwhile queues behind it at the server
(one slot, FIFO) and continues from the checkpoint the pre-heat has built so far — the server
advances its live checkpoint at every prefill chunk, so nothing is redone — which is why the
pre-heat is never killed but gated as interactive, like Ollama's. It streams because during a
prefill the server writes a client nothing but the SSE keepalive, and a failed keepalive write is
the only way it notices a client that went away. One pre-heat at a time: a prompt arriving while
one runs waits as the pending one, the latest only, and goes out when the running one ends.

Background work (photo indexing, `background=True`): the server serves one request at a time,
so an image call goes through the single-slot gate every Ds4LLM shares (`BackgroundSlot`,
Ollama's): it waits until nothing interactive is in flight and the server has been idle
`ds4.background_idle_seconds`, streams over a raw connection of its own, and a user turn arriving
mid-image shuts that socket — ds4-server marks a job cancelled on the first failed write and
stops it at the next token or prefill chunk — and takes the slot; the evicted caller gets
LLMPreempted and resends when the slot is idle again. The eviction is noticed within a token
while the image reply decodes; during its prefill the server only writes a keepalive every
5 s, so a kill that lands then takes up to that long.

The "Prefilling…" bar: ds4 says nothing on the wire until the first token (its SSE keepalive is a
bare comment), so the prefill progress is read from the server's log by the class's one
`PrefillTracker`, as Ollama's is — `ds4.log_path` (default `~/Documents/ds4/ds4-server.log`),
over ssh to `ds4.ssh_host` (default: the base_url host) when the server is on another machine.
The log says the workload at once (`prompt start` with `ctx=cached..total:remaining`) but logs a
chunk line only at the backend's chunk boundaries — a cold 26k prompt logs its first at 93%, 16 s
in —, so the bar is mostly the time estimate, refined after each reply from the usage (prompt
tokens minus the cached ones) over the prefill time the `prompt done` line reports.
"""
import json
import re
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
from .Ds4DiskCacheSweeper import Ds4DiskCacheSweeper
from .Ds4Replay import ReplayStore, replay
from .PrefillTracker import LogFormat, PrefillTracker, is_local_host, parse_last
from assistant.commands.formats import DsmlFormat as dsml
from assistant.commands.formats.CommandFormats import CommandFormats
from assistant.commands.formats.KeepFormat import KeepFormat
from common.ConfigManager import config
from common.logging import info, trace, fine, note


class Ds4LogFormat(LogFormat):
    """The prefill lines of ds4-server's log. A request opens with its workload —
    ds4-server: chat ctx=24576..26339:1763 TOOLS prompt start
    (`ctx=cached..total:remaining`) — then logs a chunk line at each backend chunk boundary, its
    numbers relative to the cached part —
    ds4-server: chat ctx=0..26339:26339 TOOLS prefill chunk 24576/26339 (93.3%) chunk=0.00 t/s avg=1518.73 t/s 16.182s
    — and closes with `prompt done 19.102s`, the prefill's seconds. A KV restore from disk logs
    the same chunk line under another phase name. The server is on one request at a time, so a
    request's lines are the section from the first `prompt start` of its window (the tracker
    has skipped the requests ahead) to the first `prompt done` after it.
    """
    _START, _DONE = 'prompt start', 'prompt done'

    def _section(self, text: str) -> tuple[str, int | None]:
        """The lines of the request the text opens with, and the index past its done line
        when that is logged."""
        start = text.find(self._START)
        if start < 0:
            return '', None
        start = text.rfind('\n', 0, start) + 1
        done = text.find(self._DONE, start)
        if done < 0:
            return text[start:], None
        end = text.find('\n', done)
        end = len(text) if end < 0 else end + 1
        return text[start:end], end

    def work(self, text: str) -> tuple[int, int] | None:
        match = re.search(r'ctx=(\d+)\.\.(\d+):(\d+)[^\n]*prompt start', self._section(text)[0])
        return (int(match.group(1)), int(match.group(3))) if match else None

    def progress(self, text: str) -> tuple[int, int] | None:
        return parse_last(r' chunk (\d+)/(\d+) \(', self._section(text)[0])

    def done(self, text: str) -> int | None:
        return self._section(text)[1]

    def duration(self, text: str) -> float | None:
        match = re.search(r'prompt done (\d+(?:\.\d+)?)s', self._section(text)[0])
        return float(match.group(1)) if match else None

    def skip(self, text: str, requests: int) -> str:
        pos = 0
        for _ in range(requests):
            done = text.find(self._DONE, pos)
            if done < 0:
                return ''
            newline = text.find('\n', done)
            pos = len(text) if newline < 0 else newline + 1
        return text[pos:]


class Ds4LLM(ChatGPTLLM):
    """A local ds4-server as an LLM backend, with DSML as the command format."""

    DEFAULT_MODEL = 'deepseek-v4.1-flash'
    DEFAULT_BASE_URL = 'http://127.0.0.1:8000/v1'
    DEFAULT_LOG_PATH = '~/Documents/ds4/ds4-server.log'   # where the server's log goes on the host that runs it
    # Measured on V4.1 Flash, cold 24k–30k prompts, 2026-10-21; each reply's usage refines it.
    _prefill = PrefillTracker("ds4", Ds4LogFormat(), seed_tps=1400.0)
    DEFAULT_REASONING_EFFORT = 'low'
    THINKING_REASONING_EFFORT = 'high'   # Thinking.FULL
    MORE_REASONING_EFFORT = 'medium'     # the floor of Thinking.MORE
    # ds4-server's spellings, lowest first; "minimal" is the smallest non-zero level, "none" alone disables reasoning.
    REASONING_EFFORTS = ('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max')
    _slot = BackgroundSlot("ds4")
    # The pre-heat guard, shared by the tiers' instances like the slot (module docstring).
    _pre_heat_lock = threading.Lock()
    _pre_heat_in_flight = False
    _pre_heat_pending: list[dict] | None = None
    PRE_HEAT_MAX_TOKENS = 1
    PRE_HEAT_READ_TIMEOUT_S = 600.0
    _logged_configs: set[tuple[str, str]] = set()
    _replay_store: ReplayStore | None = None
    CONNECT_TIMEOUT_S = 5.0

    def __init__(self, model: str | None = None, base_url: str | None = None):
        """`model` is the tier's `model:` when it has one, else `ds4.model`, else the default."""
        super().__init__(model=model or config.get('ds4.model', self.DEFAULT_MODEL), api_key_name='ds4',
                         base_url=base_url or config.get('ds4.base_url', self.DEFAULT_BASE_URL))
        if (self.model, self.base_url) not in Ds4LLM._logged_configs:
            Ds4LLM._logged_configs.add((self.model, self.base_url))
            info(f"Ds4LLM '{self.model}' using {self.base_url}, reasoning effort {self.reasoning_effort()}")
        Ds4LLM._configure_log_access(self.base_url)

    @classmethod
    def _configure_log_access(cls, base_url: str):
        """Point the prefill tracker at the server's log: the file on this machine when ds4 runs
        here, else over ssh to the serving host (`ds4.ssh_host`, default the base_url host)."""
        host = urlparse(base_url).hostname
        path = config.get('ds4.log_path', cls.DEFAULT_LOG_PATH)
        if is_local_host(host):
            cls._prefill.use_log(path)
        else:
            cls._prefill.use_log(path, ssh_host=config.get('ds4.ssh_host', host))

    def initialize(self):
        """No read timeout: a non-streamed reply says nothing until it is done, and the memory
        curation (a whole bank rewritten at effort high, ~27 t/s) runs past the SDK's 600 s —
        which then resent it, a fresh generation each time, to a server that serves one at a time."""
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                             timeout=httpx.Timeout(None, connect=self.CONNECT_TIMEOUT_S))
        trace(f"ds4 client: no read timeout, {self.CONNECT_TIMEOUT_S:.0f} s to connect")
        Ds4DiskCacheSweeper.start_once(server_is_local=is_local_host(urlparse(self.base_url).hostname))

    def _note_stop(self, finish_reason: str | None):
        """Keep the reply's stop reason where `last_stop_reason` reads it, in Anthropic's spelling
        for a reply cut at the output cap (`length` → `max_tokens`), so the curator's check sees it."""
        self._last_stop_reason = 'max_tokens' if finish_reason == 'length' else finish_reason
        if self._last_stop_reason == 'max_tokens':
            info("ds4: the reply was cut off at the output limit (finish_reason=length)")

    @property
    def api_key(self) -> str:
        """ds4-server ignores the key, but the OpenAI client refuses to start without one."""
        return 'ds4'

    def api_key_name(self) -> None:
        return None

    def missing_api_key(self) -> None:
        return None

    def name(self) -> str:
        return "DS4"

    def display_model(self) -> str:
        return self.model

    def is_local(self) -> bool:
        return True

    def default_command_format(self):
        """Keep's own call text: on V4.1 Flash it passed more of the prompt test suite than DSML,
        the syntax the model was trained on (measured 2026-10-22, see the module docstring)."""
        return KeepFormat

    def prefers_query_as_own_turn(self) -> bool:
        """The test suite's query as its own user turn: ds4-server's cold checkpoint (with
        `--kv-cache-cold-anchor last-user`, `scripts/local_ds4_start.sh`) is cut right before the last
        user turn, so a test loads the system prompt, the history fixture and the toolkits from
        disk and prefills only its query (measured 2026-10-21, see DialogManager). The V4.1
        renderer joins consecutive user-like messages — a user turn, a tool result — into one
        `<｜User｜>` block, which is why the query needs an assistant turn in front of it."""
        return True

    def prefers_prepended_user_query(self) -> bool:
        """ds4-server keeps one live KV context and reuses it only for a prompt that extends the
        previous one; a query prepended to the system prompt changes byte 0 every turn, so every
        turn re-prefilled the whole prompt (`common=2` in ds4-server.log, 2026-10-20)."""
        return False

    @classmethod
    def reasoning_effort(cls) -> str:
        """The effort of an ordinary request (thinking=False): `ds4.reasoning_effort`, or the
        default; a spelling ds4 does not know falls back to the default and is logged."""
        effort = str(config.get('ds4.reasoning_effort', cls.DEFAULT_REASONING_EFFORT)).lower()
        if effort not in cls.REASONING_EFFORTS:
            info(f"ds4.reasoning_effort '{effort}' is not one of {', '.join(cls.REASONING_EFFORTS)} — "
                 f"using {cls.DEFAULT_REASONING_EFFORT}")
            return cls.DEFAULT_REASONING_EFFORT
        return effort

    def supports_native_thinking(self) -> bool:
        return True

    def handles_thinking_internally(self) -> bool:
        """The model reasons on its own unless the effort is "none"."""
        return self.reasoning_effort() != 'none'

    def prefers_byte_prefix_tail(self) -> bool:
        # ds4 reuses its live context only for a prompt that extends it token for token (its
        # rewind is GLM-only). A trailing directive folded onto the last message is dropped when
        # the next round renders that message as history, so the prefix diverged ~20 tokens
        # before the previous prompt's end and every round fell back to the disk prefix
        # (ds4-server log: `live kv cache miss ... common=<live - 20> reason=token-mismatch`).
        return True

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
        return float(config.get('ds4.background_idle_seconds', BackgroundSlot.DEFAULT_IDLE_S))

    def background_wait_seconds(self) -> float:
        return Ds4LLM._slot.wait_seconds()

    @classmethod
    def replay_store(cls) -> ReplayStore:
        """The one side store every Ds4LLM instance fills and replays from (the tiers share
        the server and its one live context)."""
        if cls._replay_store is None:
            cls._replay_store = ReplayStore()
        return cls._replay_store

    def pre_heat_cache(self, messages: list[dict]):
        """Prefill the prompt's new prefix in the background so the next turn finds it live
        (module docstring). At most one in flight; a prompt arriving meanwhile replaces the
        pending one and goes out when the running pre-heat ends."""
        with Ds4LLM._pre_heat_lock:
            if Ds4LLM._pre_heat_in_flight:
                Ds4LLM._pre_heat_pending = messages
                trace("ds4 pre-heat deferred: one is in flight — this prompt goes out when it ends")
                return
            Ds4LLM._pre_heat_in_flight = True
        self._fire_pre_heat(messages)

    def _fire_pre_heat(self, messages: list[dict]):
        threading.Thread(target=self._run_pre_heat, args=(messages,), name="Ds4PreHeat", daemon=True).start()

    def _run_pre_heat(self, messages: list[dict]):
        try:
            self._pre_heat(messages)
        except Exception as e:
            trace(f"ds4 pre-heat failed (non-critical): {e}")
        finally:
            self._finish_pre_heat()

    def _finish_pre_heat(self):
        """Release the guard, or hand it straight to the prompt deferred meanwhile."""
        with Ds4LLM._pre_heat_lock:
            pending, Ds4LLM._pre_heat_pending = Ds4LLM._pre_heat_pending, None
            if pending is None:
                Ds4LLM._pre_heat_in_flight = False
        if pending is not None:
            info("ds4 pre-heat: sending the prompt deferred meanwhile")
            self._fire_pre_heat(pending)

    def _pre_heat(self, messages: list[dict]):
        """The request: the resolved messages as a real turn sends them (replayed assistant rows,
        the chat tier's effort, no tool), streamed, one output token. The reply is dropped — it is
        no turn, so neither the replay store nor the cache report sees it. Gated as interactive so
        a photo-indexing request waits, and shown as background on the prefill bar. Unlike a real
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
        info(f"ds4 pre-heat: prefilling the prompt's new prefix ({len(messages)} messages, "
             f"effort {kwargs['extra_body']['reasoning_effort']})...")
        started = time.monotonic()
        Ds4LLM._slot.interactive_begin()
        token = self._prefill.begin(interactive=False)
        try:
            usage = None
            for chunk in client.chat.completions.create(**kwargs):
                if getattr(chunk, 'usage', None) is not None:
                    usage = chunk.usage
            # Read before `end` drops the request from the tracker.
            seconds = self._prefill.prefill_seconds(token) if usage is not None else None
        finally:
            self._prefill.end(token)
            Ds4LLM._slot.interactive_end()
        elapsed = time.monotonic() - started
        if usage is None:
            info(f"ds4 pre-heat complete in {elapsed:.1f} s (the server reported no usage)")
            return
        details = getattr(usage, 'prompt_tokens_details', None)
        cached = getattr(details, 'cached_tokens', None) if details is not None else None
        prefilled = usage.prompt_tokens - (cached or 0)
        info(f"ds4 pre-heat complete: {usage.prompt_tokens} tokens, {cached if cached is not None else '?'} cached, "
             f"{prefilled} prefilled in {elapsed:.1f} s")
        if seconds:
            self._prefill.observe_rate(prefilled, seconds)

    def _request_kwargs(self, messages: list[dict], thinking: Thinking | bool, stream: bool) -> dict:
        messages = replay(self._convert_anthropic_content(messages), self.replay_store())
        # No `tools`: see the module docstring — a declared tool costs the live cache and pulls the model's calls to it.
        kwargs = dict(model=self.model, messages=messages, stream=stream,
                      extra_body={'reasoning_effort': self._effort_for(thinking)})
        if stream:
            kwargs['stream_options'] = {'include_usage': True}
        return kwargs

    @staticmethod
    def _log_usage(usage, started: float, reasoning_chars: int = 0) -> int | None:
        """One line per request: prompt tokens with the server's cache hits, output tokens, time.
        Returns the tokens the server prefilled (prompt minus cached), None when it said nothing."""
        if usage is None:
            trace("ds4: response carried no usage")
            return None
        details = getattr(usage, 'prompt_tokens_details', None)
        cached = getattr(details, 'cached_tokens', None) if details is not None else None
        cached_text = f", {cached} cached" if cached is not None else ""
        thought = f", {reasoning_chars} reasoning chars" if reasoning_chars else ""
        info(f"ds4 metrics: ↑ {usage.prompt_tokens} tokens{cached_text} | ↓ {usage.completion_tokens} tokens{thought} "
             f"| {time.monotonic() - started:.2f}s")
        return usage.prompt_tokens - (cached or 0)

    def cache_scope(self) -> str | None:
        return f"ds4 {self.base_url}"

    def _cache_report(self, messages: list[dict], effort: str):
        """One request's cache-break report, made once: early, from the workload line the
        server logs as the prefill starts — the Home tab's row then lands while the bar is
        still drawn — or, when the log is unreachable, at the end from the reply's usage.
        (early, late): `early` is the tracker's `on_work` callback, `late` takes the usage and
        reports only when `early` never ran. The two can race on a poll that finishes as the
        stream ends, hence the lock."""
        lock, made_from = threading.Lock(), []

        def once(cached: int, prompt_tokens: int, source: str):
            with lock:
                if made_from:
                    trace(f"ds4 cache: the {source} says {cached}/{prompt_tokens} cached — "
                          f"already reported from the {made_from[0]}")
                    return
                made_from.append(source)
            self.record_cache_hit(cached, prompt_tokens)
            self._report_cache(messages, cached, prompt_tokens, effort)

        def early(cached: int, remaining: int):
            once(cached, cached + remaining, "workload line")

        def late(usage):
            details = getattr(usage, 'prompt_tokens_details', None) if usage is not None else None
            cached = getattr(details, 'cached_tokens', None) if details is not None else None
            if cached is None:
                trace("ds4 cache: the reply carries no cached-token count — no report")
                return
            once(cached, usage.prompt_tokens, "usage")

        return early, late

    def _report_cache(self, messages: list[dict], cached: int, prompt_tokens: int, effort: str):
        """The cached tokens against the prompt, for the cache-break report. The server renders
        past turns by the reasoning effort, so an effort other than the previous request's is
        the note the report falls back on when the prompt text is unchanged."""
        previous, self._last_effort = getattr(self, '_last_effort', None), effort
        note = None
        if previous is not None and previous != effort:
            note = f"the reasoning effort changed from {previous} to {effort}, which re-renders the past turns"
        self.report_cache_usage(messages, cached, prompt_tokens, note=note)

    # --- the DSML side ----------------------------------------------------------------------

    @staticmethod
    def _rendered_calls(calls) -> list[tuple[str, object]]:
        """Parsed tool calls (the merged stream deltas, or a message's `tool_calls`) each
        written back as call text in the active command format — DSML under a ds4 smart tier,
        the smart tier's syntax otherwise, so the parser reads it — paired with the call; an
        entry without a name is dropped."""
        fmt = CommandFormats.active()
        rendered = []
        for call in calls:
            function = getattr(call, 'function', None)
            name = getattr(function, 'name', None)
            if not name:
                trace(f"ds4 tool call without a function name, dropped: {call!r}")
                continue
            arguments = getattr(function, 'arguments', None) or "{}"
            trace(f"ds4 parsed a {name} call — written back as {fmt.name} call text")
            rendered.append((fmt.render_tool_call(name, arguments), call))
        return rendered

    @staticmethod
    def _tail(rendered: list[tuple[str, object]]) -> str:
        return "\n".join(text for text, _ in rendered)

    def _closing(self, reply: str, withheld: str, rendered: list[tuple[str, object]]) -> tuple[str, str]:
        """What follows the reply text — the parsed calls as call text on a line of their own, or
        the withheld block released as written — and the content as the model wrote it, for the
        replay store (before a parsed call: the text the server kept, trailing whitespace trimmed)."""
        if rendered:
            tail = self._tail(rendered)
            return ("\n" + tail if reply and not reply.endswith("\n") else tail), reply.rstrip()
        released = self._release_withheld(withheld)
        return released, reply + released

    def _remember(self, reply: str, reasoning: str, verbatim: str):
        """Give the replay store what the next request needs to send this reply back as the
        model wrote it: `reply` as Keep's history will hold it, the reasoning, the content."""
        self.replay_store().remember_reply(reply, reasoning, verbatim)

    @staticmethod
    def _split_at_sentinel(text: str) -> tuple[str, str]:
        """`text` cut before the first DSML opener: the reply text to show, and the block text
        the server parses for us (empty when the text holds none)."""
        at = text.find(dsml.OPEN)
        return (text, "") if at < 0 else (text[:at], text[at:])

    @staticmethod
    def _release_withheld(withheld: str) -> str:
        """The content withheld from the DSML sentinel on, once the reply is complete and no
        parsed call was rendered in its place: the text as written — Keep's own parser then
        reads what it can. The caller passes it only in that case: a `tool_calls` delta
        without a function name renders to nothing, and the text it stood for must not be
        lost with it."""
        if not withheld:
            return ""
        if CommandFormats.active() is dsml.DsmlFormat:
            trace(f"ds4: a DSML block released whole to the command parser ({withheld.count(dsml.OPEN)} opener(s))")
            return withheld
        note(f"ds4: the model wrote a DSML block although the prompt's syntax is {CommandFormats.active().name} "
             f"(or the server handed back a call without a name); the rescue reads it as written "
             f"({withheld.count(dsml.OPEN)} opener(s))")
        return withheld

    def create_streaming_completion(self, messages: list[dict], thinking: Thinking | bool = Thinking.NORMAL,
                                    volatile_tail: int = 0, follow_up: bool = False) -> Generator[str, None, None]:
        """Reasoning deltas (`reasoning_content`) are yielded wrapped in <thinking> tags, the reply
        text after them up to the first DSML sentinel; from there the content is withheld and
        the `tool_calls` deltas (merged by index) are yielded as call text in the active format
        at the end. The reply then goes to the replay store, and the stream's closing usage
        chunk is logged."""
        self._ensure_initialized()
        started = time.monotonic()
        Ds4LLM._slot.interactive_begin()
        early, late = self._cache_report(messages, self._effort_for(thinking))
        token = self._prefill.begin(on_work=early)
        try:
            yield from self._stream(messages, thinking, started, token, late)
        finally:
            self._prefill.end(token)
            Ds4LLM._slot.interactive_end()

    def _stream(self, messages: list[dict], thinking: Thinking | bool, started: float, token: int | None,
                report_cache) -> Generator[str, None, None]:
        self._last_stop_reason = None
        response = self.client.chat.completions.create(**self._request_kwargs(messages, thinking, stream=True))
        in_thinking, reasoning, usage = False, "", None
        reply, pending, withheld = "", "", ""  # text yielded; text held for a split opener; the block text
        calls: dict[int, _ToolCall] = {}
        prefilling = True
        for chunk in response:
            if getattr(chunk, 'usage', None) is not None:
                usage = chunk.usage
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
                if withheld:
                    withheld += delta.content
                else:
                    visible, withheld = self._split_at_sentinel(pending + delta.content)
                    if withheld:
                        pending = ""
                    else:
                        visible, pending = self._hold_opener_prefix(visible)
                    if visible:
                        reply += visible
                        yield visible
            for entry in getattr(delta, 'tool_calls', None) or []:
                index = getattr(entry, 'index', None) or 0
                calls.setdefault(index, _ToolCall()).merge(entry)
        if in_thinking:
            trace("ds4 stream ended inside the thinking block — closing it")
            yield "</thinking>"
        if pending:
            reply += pending
            yield pending
        if withheld:
            trace(f"ds4 withheld {len(withheld)} chars of DSML text from the reply ({len(calls)} tool call(s) parsed)")
        rendered = self._rendered_calls([calls[i] for i in sorted(calls)])
        tail, verbatim = self._closing(reply, withheld, rendered)
        if tail:
            yield tail
        self._remember(reply + tail, reasoning, verbatim)
        prefilled = self._log_usage(usage, started, len(reasoning))
        report_cache(usage)
        if prefilled is not None:
            seconds = self._prefill.prefill_seconds(token)
            if seconds:
                self._prefill.observe_rate(prefilled, seconds)

    @staticmethod
    def _hold_opener_prefix(text: str) -> tuple[str, str]:
        """`text` with any suffix that is a prefix of the DSML opener held back, so an opener
        split over two deltas is still seen whole: (to yield, to hold)."""
        for length in range(min(len(dsml.OPEN) - 1, len(text)), 0, -1):
            if text.endswith(dsml.OPEN[:length]):
                return text[:-length], text[-length:]
        return text, ""

    def create_completion(self, messages: list[dict], thinking: Thinking | bool = Thinking.NORMAL, volatile_tail: int = 0,
                          follow_up: bool = False) -> str:
        """The reply text, then every parsed tool call written back as DSML text; the model's
        reasoning is logged, not returned."""
        self._ensure_initialized()
        started = time.monotonic()
        Ds4LLM._slot.interactive_begin()
        early, late = self._cache_report(messages, self._effort_for(thinking))
        token = self._prefill.begin(on_work=early)
        self._last_stop_reason = None
        try:
            response = self.client.chat.completions.create(**self._request_kwargs(messages, thinking, stream=False))
        finally:
            self._prefill.end(token)
            Ds4LLM._slot.interactive_end()
        self._note_stop(response.choices[0].finish_reason)
        message = response.choices[0].message
        reasoning = getattr(message, 'reasoning_content', None) or ""
        if reasoning:
            fine(f"ds4 thought:\n{reasoning.strip()}")
        self._log_usage(getattr(response, 'usage', None), started, len(reasoning))
        late(getattr(response, 'usage', None))
        calls = getattr(message, 'tool_calls', None) or []
        content, withheld = self._split_at_sentinel(message.content or "")
        rendered = self._rendered_calls(calls)
        tail, verbatim = self._closing(content, withheld, rendered)
        self._remember(content + tail, reasoning, verbatim)
        return content + tail

    # --- images -----------------------------------------------------------------------------

    def create_completion_with_images(self, messages: list[dict], images: list, background: bool = False) -> str:
        """A background image request (photo indexing) goes through the slot gate over a raw
        connection the gate can shut; an interactive one is the plain call, evicting any
        background request first."""
        if background:
            return self._background_completion(self._image_messages(messages, images))
        Ds4LLM._slot.interactive_begin()
        try:
            return super().create_completion_with_images(messages, images)
        finally:
            Ds4LLM._slot.interactive_end()

    def _background_completion(self, messages: list[dict]) -> str:
        """The abortable image request: waits for the idle slot, streams the answer over its
        own connection (SSE, the OpenAI shape: `data: {chunk}` lines, `data: [DONE]` last, a
        bare `: prefill` comment as the keepalive while the server prefills) and raises
        LLMPreempted when a user request evicts it. No tool is declared: an image request
        matches no live context anyway, and its reply is text."""
        body = json.dumps({'model': self.model, 'messages': messages, 'stream': True,
                           'stream_options': {'include_usage': True},
                           'reasoning_effort': self.reasoning_effort()})
        path = urlparse(self.base_url).path.rstrip('/') + '/chat/completions'
        slot = Ds4LLM._slot
        slot.acquire(self.background_idle_seconds())
        started = time.monotonic()
        early, late = self._cache_report(messages, self.reasoning_effort())
        token = self._prefill.begin(interactive=False, on_work=early)
        chunks: list[str] = []
        reasoning_chars, usage = 0, None
        try:
            conn = slot.open_connection(self.base_url)
            slot.register_connection(conn)
            conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            if resp.status != 200:
                raise RuntimeError(f"ds4 {path} returned {resp.status}: {resp.read(500)!r}")
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
                for choice in data.get('choices') or []:
                    delta = choice.get('delta') or {}
                    if delta.get('reasoning_content'):
                        self._prefill.first_token(token)
                        reasoning_chars += len(delta['reasoning_content'])
                    if delta.get('content'):
                        self._prefill.first_token(token)
                        chunks.append(delta['content'])
            else:
                raise RuntimeError("ds4 closed the stream before the response was done")
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
