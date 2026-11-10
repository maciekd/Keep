"""Talks to a local ds4-server through its OpenAI-compatible API and runs Keep's commands in Keep's own call text.

ds4 (antirez's DwarfStar engine) serves one large open-weight model — DeepSeek V4.1 Flash by
default — resident on this machine's GPU, at `ds4.base_url` (default http://127.0.0.1:8000/v1);
`model` is the id the server loaded (`/v1/models`). The OpenAI side — the client, the stream,
the reasoning effort from the request's `Thinking` (`ds4.reasoning_effort` for a turn, default
"low" so it answers in seconds), the metrics line, the cache report, the pre-heat and the
single-slot gate — is `OpenAICompatibleLLM`, which this class is the ds4 flavour of: what is
below is what ds4-server does differently from any other local server. The autonomous check's
higher effort makes its prompt share the chat's cached prefix only up to the first assistant
turn of the history, since the effort changes how the server renders past turns.

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
bare comment, and it knows no `return_progress`), so the prefill progress is read from the
server's log by the class's one `PrefillTracker`, as Ollama's is — `ds4.log_path` (default
`~/Documents/ds4/ds4-server.log`), over ssh to `ds4.ssh_host` (default: the base_url host) when
the server is on another machine.
The log says the workload at once (`prompt start` with `ctx=cached..total:remaining`) but logs a
chunk line only at the backend's chunk boundaries — a cold 26k prompt logs its first at 93%, 16 s
in —, so the bar is mostly the time estimate, refined after each reply from the usage (prompt
tokens minus the cached ones) over the prefill time the `prompt done` line reports.
"""
import re
import threading
from urllib.parse import urlparse

from .Ds4DiskCacheSweeper import Ds4DiskCacheSweeper
from .Ds4Replay import ReplayStore, replay
from .OpenAICompatibleLLM import OpenAICompatibleLLM, _Content
from .PrefillTracker import LogFormat, PrefillTracker, is_local_host, parse_last
from .BackgroundSlot import BackgroundSlot
from assistant.commands.formats import DsmlFormat as dsml
from assistant.commands.formats.CommandFormats import CommandFormats
from assistant.commands.formats.KeepFormat import KeepFormat
from common.ConfigManager import config
from common.logging import trace, note


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


class _DsmlContent(_Content):
    """The reply text up to the first DSML sentinel is visible; from there the content is
    withheld, whole, for the end of the reply (module docstring). An opener split over two
    deltas is still seen whole: a suffix that is a prefix of the opener is held back."""

    def __init__(self):
        self.pending = ""    # text held for a split opener
        self.withheld = ""   # the block text, from the first sentinel on

    def take(self, text: str) -> str:
        if self.withheld:
            self.withheld += text
            return ""
        visible, self.withheld = self.split(self.pending + text)
        if self.withheld:
            self.pending = ""
            return visible
        visible, self.pending = self._hold_opener_prefix(visible)
        return visible

    def flush(self) -> str:
        pending, self.pending = self.pending, ""
        return pending

    @staticmethod
    def split(text: str) -> tuple[str, str]:
        """`text` cut before the first DSML opener: the reply text to show, and the block text
        the server parses for us (empty when the text holds none)."""
        at = text.find(dsml.OPEN)
        return (text, "") if at < 0 else (text[:at], text[at:])

    @staticmethod
    def _hold_opener_prefix(text: str) -> tuple[str, str]:
        for length in range(min(len(dsml.OPEN) - 1, len(text)), 0, -1):
            if text.endswith(dsml.OPEN[:length]):
                return text[:-length], text[-length:]
        return text, ""


class Ds4LLM(OpenAICompatibleLLM):
    """A local ds4-server as an LLM backend: the OpenAI-compatible backend plus ds4's replay
    of the history, its DSML block and its log-fed prefill bar."""

    SECTION = 'ds4'
    NAME = "DS4"
    LABEL = "ds4"
    DEFAULT_MODEL = 'deepseek-v4.1-flash'
    DEFAULT_LOG_PATH = '~/Documents/ds4/ds4-server.log'   # where the server's log goes on the host that runs it
    RETURN_PROGRESS = False   # ds4-server streams no progress; the bar reads its log
    # Measured on V4.1 Flash, cold 24k–30k prompts, 2026-10-21; each reply's usage refines it.
    _prefill = PrefillTracker("ds4", Ds4LogFormat(), seed_tps=1400.0)
    # ds4-server's spellings, lowest first; "minimal" is the smallest non-zero level, "none" alone disables reasoning.
    REASONING_EFFORTS = ('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max')
    _slot = BackgroundSlot("ds4")
    _pre_heat_lock = threading.Lock()
    _pre_heat_in_flight = False
    _pre_heat_pending: list[dict] | None = None
    _logged_configs: set[tuple[str | None, str]] = set()
    _models_checked: set[tuple[str, str]] = set()
    _discovered_models: dict[str, str] = {}
    _discovery_failed: set[str] = set()
    _replay_store: ReplayStore | None = None

    def _configure_prefill(self):
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
        super().initialize()
        Ds4DiskCacheSweeper.start_once(server_is_local=is_local_host(urlparse(self.base_url).hostname))

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

    def resumes_thinking_after_calls(self) -> bool:
        """Seen four times on prod (2026-11-05, 2026-11-08), always right after a call: the
        server routes only the first thinking block to `reasoning_content`, and its guard
        for a second one is armed only for a request that declares tools — Keep never does."""
        return self.reasoning_effort() != 'none'

    def prefers_byte_prefix_tail(self) -> bool:
        # ds4 reuses its live context only for a prompt that extends it token for token (its
        # rewind is GLM-only). A trailing directive folded onto the last message is dropped when
        # the next round renders that message as history, so the prefix diverged ~20 tokens
        # before the previous prompt's end and every round fell back to the disk prefix
        # (ds4-server log: `live kv cache miss ... common=<live - 20> reason=token-mismatch`).
        return True

    # --- the replay ---------------------------------------------------------------------------

    @classmethod
    def replay_store(cls) -> ReplayStore:
        """The one side store every Ds4LLM instance fills and replays from (the tiers share
        the server and its one live context)."""
        if cls._replay_store is None:
            cls._replay_store = ReplayStore()
        return cls._replay_store

    def _prepare_messages(self, messages: list[dict]) -> list[dict]:
        return replay(messages, self.replay_store())

    def _remember(self, reply: str, reasoning: str, verbatim: str):
        """Give the replay store what the next request needs to send this reply back as the
        model wrote it: `reply` as Keep's history will hold it, the reasoning, the content."""
        self.replay_store().remember_reply(reply, reasoning, verbatim)

    # --- the DSML side ----------------------------------------------------------------------

    def _content(self) -> _Content:
        return _DsmlContent()

    @staticmethod
    def _split_at_sentinel(text: str) -> tuple[str, str]:
        return _DsmlContent.split(text)

    @staticmethod
    def _hold_opener_prefix(text: str) -> tuple[str, str]:
        return _DsmlContent._hold_opener_prefix(text)

    def _release_withheld(self, withheld: str) -> str:
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
