import time
from typing import Generator

import httpx

from .AbstractLLM import AbstractLLM, LLMRefusalError, api_error_message, Thinking
from anthropic import Anthropic, APIConnectionError, APIStatusError
from common.ConfigManager import config
from common.DevelopmentConfig import BYTE_PREFIX_TAIL, CLAUDE_STATIC_CACHE_TTL, CLAUDE_HISTORY_CACHE_TTL
from common.logging import trace, info


WORKSPACE_CONFIG_KEY = "api_keys.anthropic_workspace"
WORKSPACE_HEADER = "anthropic-workspace-id"
WORKSPACE_REQUIRED = (
    "The Anthropic key in config.yaml has its Scope set to Organization, so Anthropic needs to be "
    "told which workspace to use. Either create a key at https://platform.claude.com/settings/keys "
    "with Scope set to Default workspace (and Expires to Never), then put it in api_keys.anthropic — "
    f"or keep this key and set {WORKSPACE_CONFIG_KEY} to the id of a workspace (wrkspc_…, the ID "
    "column at https://platform.claude.com/settings/workspaces).")


class WorkspaceRequiredError(RuntimeError):
    """Anthropic refused every request because the key names no workspace — WORKSPACE_REQUIRED
    says what to do, where the server's own line only names the missing header."""


def claude_workspace_id() -> str | None:
    """The workspace a key not tied to one acts in, or None: a key created for one workspace
    needs nothing, and the config key then stays empty."""
    if config.missing_required((WORKSPACE_CONFIG_KEY,)):
        return None
    return config.get(WORKSPACE_CONFIG_KEY).strip()


def claude_default_headers() -> dict[str, str] | None:
    """The headers every Anthropic client of ours sends: the workspace id when one is configured."""
    workspace = claude_workspace_id()
    return {WORKSPACE_HEADER: workspace} if workspace else None


def workspace_required(e: Exception) -> bool:
    """Whether *e* is Anthropic's 400 for a key not tied to a workspace sent without the header."""
    return getattr(e, "status_code", None) == 400 and WORKSPACE_HEADER in (api_error_message(e) or "")


class ClaudeLLM(AbstractLLM):
    """Talks to Anthropic's Messages API with prompt caching, extended thinking and fast mode.

    Places cache_control breakpoints from the prompt's CACHE_BREAKPOINT_MARKERs (3 system + 1
    history — on the last message every later request sends unchanged, see _message_cache_index;
    a follow-up round of a request caches its whole tail for FOLLOW_UP_CACHE_TTL instead),
    retries retryable status errors and raises LLMRefusalError when the safety layer keeps refusing.
    """

    FAST_MODE_BETA = "fast-mode-2026-02-01"
    REFUSAL_FALLBACK_MODEL = "claude-opus-5"
    REFUSAL_FALLBACK_CACHE_TTL = "5m"
    # The families that think adaptively when the `thinking` parameter is omitted. Opus 4.x,
    # Sonnet 4.x and Haiku 4.5 do not: without the parameter they do not think at all.
    ALWAYS_THINKING_MODEL_PREFIXES = ("claude-fable", "claude-mythos", "claude-opus-5", "claude-sonnet-5")

    # The models whose chat requests keep the prompt prefix cacheable across requests: no user
    # query prepended to the top of the prompt (prefers_prepended_user_query) and a byte-stable
    # trailing turn.
    PREFIX_CACHED_MODEL_PREFIXES = ("claude-fable", "claude-opus-5-5")
    # True while the coming request's prefix is shared by no later request (the user query
    # prepended at byte 0, RequestProcessor → set_per_request_prefix): every block then takes
    # FOLLOW_UP_CACHE_TTL, since only the request's own follow-up rounds, seconds later, read
    # it — a 1h write (2x) would die unread where a 5m one (1.25x) serves the same reads. The
    # autonomous check never prepends and has an LLM instance of its own, so its hourly checks
    # keep reading each other's 1h blocks.
    per_request_prefix = False

    # The contract: every Claude request runs at high effort. The API's own default is high
    # on most models but medium on Opus 5.5 and Haiku 5.5, so it is sent explicitly on the
    # families that take the parameter. Constant across a conversation — a change mid-dialog
    # would invalidate the message cache.
    EFFORT = "high"
    EFFORT_MODEL_PREFIXES = ALWAYS_THINKING_MODEL_PREFIXES + ("claude-haiku-5",)

    # CACHE_BREAKPOINT_MARKER is inherited from AbstractLLM.
    # Anthropic allows 4 cache_control breakpoints per request; _build_api_params
    # reserves one for the message-history breakpoint, leaving 3 for the system
    # prompt. Markers beyond this limit are ignored (their text joins the tail).
    MAX_SYSTEM_CACHE_BLOCKS = 3
    # TTL of the history breakpoint on a follow-up round (one that ends in a command result).
    # The next round of the same request repeats the tail verbatim seconds later, and no request
    # after that one does: the window is re-rendered between requests (an oversized result cut
    # to its history copy, the clock note moved), so the next check 55 minutes on reads the
    # first round's block and never a later round's (prod 2026-10-19: every check). A 5m write
    # costs 1.25x input, the 1h write 2x, and the read within the round is the same 0.1x either
    # way; prod wrote ~340K tokens a day of follow-up tails at the 1h tier.
    FOLLOW_UP_CACHE_TTL = "5m"
    # The TTLs Anthropic offers, shortest first: a request's breakpoints may only get shorter.
    CACHE_TTL_RANK = {"5m": 0, "1h": 1}
    # Prompt test suite only — None in production. The text of the toolkit-result turn the
    # suite appends to its fixture history (PromptManager._all_toolkits_loaded): every test
    # resets the dialog, so the query is merged into that turn and the history breakpoint on
    # it wrote the fixture, the toolkits and the query anew for every test — ~16K tokens a
    # test, read by nobody (69.6% hit rate on 178 tests, 2026-10-21). A user turn starting
    # with this text is split in two blocks with a breakpoint on the first, so the stable
    # part is read from the second test on. Set by the suite's prompt loader, never by the
    # server.
    TEST_SUITE_CACHED_TAIL: str | None = None
    _ttl_cuts_logged: set[tuple[str, str]] = set()  # the pool builds one LLM per handler: say it once

    # Output caps. With adaptive thinking the thinking tokens count against max_tokens too,
    # so a thinking call that must emit a long answer (the memory curator rewrites a ~8k-token
    # bank) needs real headroom: at 16000 the curator's replies were cut mid-sentence for weeks,
    # and at 32000 a run whose thinking ran long was still cut in the bank's last section.
    THINKING_MAX_TOKENS = 64000
    # The chat cap. The 5-generation models think without the parameter and the thinking
    # counts here too: at 4096, Opus 5.5 at high effort thought through the whole cap three
    # times running on a home-automation plan and the user got silence (issue #108). The
    # largest chat output seen on prod before that was 3920 tokens; the cap costs nothing
    # unless used, and a round that hits it is told so and gets one more (RequestProcessor).
    MAX_TOKENS = 16000
    # The SDK refuses a non-streaming request whose max_tokens implies more than 10 minutes
    # (about 21k tokens) unless a timeout is given explicitly. 32000 tokens took ~7 minutes
    # at the rate the nightly curation has shown, so 30 minutes covers the cap with headroom.
    NON_STREAMING_TIMEOUT = httpx.Timeout(30 * 60, connect=5.0)

    def __init__(self, model: str = "claude-sonnet-4-5", speed: str | None = None, prompt_cache: bool = True,
                 static_cache_ttl: str | None = None, history_cache_ttl: str | None = None):
        self.model = model
        self.speed = speed
        self.prompt_cache = prompt_cache
        self.static_cache_ttl = static_cache_ttl or CLAUDE_STATIC_CACHE_TTL
        self.history_cache_ttl = history_cache_ttl or CLAUDE_HISTORY_CACHE_TTL
        if self.CACHE_TTL_RANK[self.history_cache_ttl] > self.CACHE_TTL_RANK[self.static_cache_ttl]:
            # Anthropic rejects a longer TTL after a shorter one, and the history breakpoint comes
            # after the system blocks — the prompt test runner shortens the static TTL alone.
            if (self.history_cache_ttl, self.static_cache_ttl) not in ClaudeLLM._ttl_cuts_logged:
                ClaudeLLM._ttl_cuts_logged.add((self.history_cache_ttl, self.static_cache_ttl))
                info(f"Claude cache: history TTL {self.history_cache_ttl} cut to the static TTL "
                     f"{self.static_cache_ttl} — a breakpoint never outlives the blocks before it")
            self.history_cache_ttl = self.static_cache_ttl
        self.client = None
        self._last_stop_reason = None
        self._cancelled = False

    def name(self) -> str:
        suffix = " (fast)" if self.speed == "fast" else ""
        return f"Claude ({self.model}){suffix}"

    @property
    def api_key(self) -> str | None:
        """Read at every use so a key stored at runtime reaches the next request (see GeminiLLM)."""
        return config.get_api_key('anthropic')

    def api_key_name(self) -> str:
        return "api_keys.anthropic"

    def missing_api_key(self) -> str | None:
        return None if self.api_key else self.api_key_name()

    def provider(self) -> str:
        return "Claude"

    def display_model(self) -> str:
        """"Fable" from claude-fable-5-1, "Opus" from claude-opus-5 — the family, spoken."""
        parts = self.model.split("-")
        return parts[1].capitalize() if len(parts) > 1 else self.model

    def refusal_fallback(self) -> AbstractLLM | None:
        """Fable's dual-use classifier refuses ordinary home-automation turns (e.g. any
        request right after a "server's just come up, nothing configured" reply) and does
        so deterministically, so re-sending never helps. Opus answers the same request;
        hand it over for this one turn. Refusals come in bursts — the turns following a
        refused one tend to be refused too — so the fallback caches its prompt like the
        primary does, but with the 5-minute TTL: the first fallback of a burst pays the
        1.25x write instead of the 1h tier's 2x, every later one within the burst reads
        at 0.1x, and a lone refusal wastes only a quarter of its input. The prompt is
        built for the primary before the fallback is chosen, so its cached prefix is the
        same bytes on every fallback. Opus 5 does have a fast mode, but at twice the
        price; a refused turn is already being paid for twice, so the speed flag is not
        carried over."""
        if not self.model.startswith("claude-fable"):
            return None
        return ClaudeLLM(self.REFUSAL_FALLBACK_MODEL, static_cache_ttl=self.REFUSAL_FALLBACK_CACHE_TTL)

    def supports_native_thinking(self) -> bool:
        """Claude supports native thinking via extended thinking API."""
        return True

    def handles_thinking_internally(self) -> bool:
        """On the 5-generation families adaptive thinking runs whether or not the `thinking`
        parameter is sent (Fable 5.1 cannot even turn it off), and asking for a <thinking> tag
        on top gets the request refused as reasoning extraction. An older model (the
        constructor default included) only thinks when asked, so it keeps the tag instruction
        — without it it would not reason at all."""
        return self.model.startswith(self.ALWAYS_THINKING_MODEL_PREFIXES)

    def refuses_reasoning_requests(self) -> bool:
        """The same families: Sonnet 5.5, Opus 5.5 and Fable 5.1 answer a prompt that asks for a
        "thinking" field with `stop_reason: refusal`, category `reasoning_extraction`, and no text
        at all (measured 2026-10-19 on the photo-indexing prompt); Haiku 5.5 answers it."""
        return self.model.startswith(self.ALWAYS_THINKING_MODEL_PREFIXES)

    def prefers_prepended_user_query(self) -> bool:
        """Prepending the user query to the top of the system prompt puts per-request
        content at byte 0 of the cached prefix: every request writes the whole prefix anew
        and only its own follow-up rounds read it (per_request_prefix), never the next
        request or the autonomous check. It nudges accuracy up, so it stays on for older
        Claude models — but on Fable and Opus 5.5 the cache economics are the steepest (1h
        writes at 2x, reads at 0.025x and 0.05x) and prompt scaffolding helps least, so they
        opt out and let the shared prefix cache work."""
        return not self.model.startswith(self.PREFIX_CACHED_MODEL_PREFIXES)

    def prefers_byte_prefix_tail(self) -> bool:
        """A model whose chat requests use the prefix cache gets its trailing turn sent as the
        history renders it afterwards, so the cache breakpoint can sit on that turn
        (_message_cache_index) and the next round reads it. With the per-request directives
        folded on, the turn changed between rounds: it went out uncached and was then written
        to the cache a round later — paid for twice. Models that get the prepended query have
        no prefix cache to protect, and one that needs the <thinking> tag instruction would
        lose it in this mode, so both keep the old tail."""
        return self.handles_thinking_internally() and not self.prefers_prepended_user_query()

    def set_per_request_prefix(self, per_request: bool):
        """See per_request_prefix. Sticky instance state like set_prompt_cache: RequestProcessor
        sets it on every dialog request from whether that request prepends the user query."""
        self.per_request_prefix = per_request
        if per_request:
            trace(f"Claude cache: the prefix is this request's own — every block cached for {self.FOLLOW_UP_CACHE_TTL}")

    def _message_cache_index(self, volatile_tail: int = 0, follow_up: bool = False) -> int:
        """Which message carries the history cache breakpoint: the last one when the trailing
        turn is byte-stable (the next round's prefix then ends exactly there), else the one
        before it — a trailing turn that will be re-rendered would only ever be written.
        `volatile_tail` trailing messages the caller folded on for this call alone (the
        autonomous check's instruction, which no later request sends) move it further back
        the same way. Anthropic has a cache entry only where a request put a breakpoint, so a
        breakpoint on bytes no later request repeats leaves the next one nothing to read: it
        rewrites the whole window (12–22K tokens on prod, 2026-10-10..12, after every periodic
        check). A follow-up round ignores the volatile tail: its tail is a command result the
        next round of the same request repeats verbatim, so it is cached whole — at
        FOLLOW_UP_CACHE_TTL, since nothing after that request reads it. Moving the breakpoint
        in front of an oversized result instead (2026-10-18) made every later round of the
        request re-send the whole tail uncached: one 13-round question re-sent 32–78K tokens
        ten times, $2.4 of a $3 turn (prod 2026-10-20)."""
        stable = -1 if BYTE_PREFIX_TAIL and self.prefers_byte_prefix_tail() else -2
        if follow_up:
            return stable
        return min(stable, -1 - volatile_tail)

    @staticmethod
    def _carries_image(message: dict) -> bool:
        content = message.get('content')
        return isinstance(content, list) and any(
            isinstance(block, dict) and block.get('type') == 'image' for block in content)

    def set_prompt_cache(self, enabled: bool):
        """Toggle Anthropic prompt caching for the next request. Prepending the user
        query (see prefers_prepended_user_query) puts per-request content at byte 0 of
        the cached prefix, so caching there only ever writes and never reads — pure
        waste when nothing later reads it — the memory condenser turns the cache on for its
        run over one bank and off after. Note this is sticky instance state: any other
        consumer reusing this instance inherits the last value set (background agents are
        unaffected — they create their own instances, which default to caching on)."""
        self.prompt_cache = enabled

    def initialize(self):
        """Initialize the Anthropic client"""
        headers = claude_default_headers()
        if headers:
            info(f"Claude client sends {WORKSPACE_HEADER} {headers[WORKSPACE_HEADER]} on every request")
        self.client = Anthropic(api_key=self.api_key, default_headers=headers)

    def _convert_messages(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Convert OpenAI message format to Anthropic format.

        Returns:
            Tuple of (system_message, anthropic_messages)
        """
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            elif msg['role'] in ('user', 'assistant'):
                anthropic_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })

        return system_message, anthropic_messages

    def _build_cached_system(self, system_message: str) -> list[dict]:
        """Build the `system` field as Anthropic content blocks with cache breakpoints.

        The system message may contain up to MAX_SYSTEM_CACHE_BLOCKS
        CACHE_BREAKPOINT_MARKERs, ordered stable → volatile. Each marker ends a
        cached block at self.static_cache_ttl (default 1h; FOLLOW_UP_CACHE_TTL on a
        per_request_prefix request, whose blocks no later request reads). A breakpoint caches
        the cumulative prefix up to itself, so when a segment's bytes change only
        that segment's block and the ones after it re-write — earlier breakpoints
        still read. Text after the last marker is the uncached volatile tail
        (e.g. per-request instructions, time). Without any marker, fall back to a
        single 5-minute cached block of the whole system message.
        """
        if self.CACHE_BREAKPOINT_MARKER not in system_message:
            return [{"type": "text", "text": system_message, "cache_control": {"type": "ephemeral"}}]

        segments = system_message.split(self.CACHE_BREAKPOINT_MARKER)
        cached_segments, tail_segments = segments[:-1], [segments[-1]]
        if len(cached_segments) > self.MAX_SYSTEM_CACHE_BLOCKS:
            # Keep the FIRST markers — the prompt orders content stable → volatile,
            # so early breakpoints guard the segments most likely to stay byte-identical.
            info(f"System prompt has {len(cached_segments)} cache breakpoints, "
                 f"only the first {self.MAX_SYSTEM_CACHE_BLOCKS} are used — "
                 f"the remaining segments join the uncached tail")
            tail_segments = cached_segments[self.MAX_SYSTEM_CACHE_BLOCKS:] + tail_segments
            cached_segments = cached_segments[:self.MAX_SYSTEM_CACHE_BLOCKS]

        ttl = self.FOLLOW_UP_CACHE_TTL if self.per_request_prefix else self.static_cache_ttl
        blocks = []
        for segment in cached_segments:
            text = segment.strip()
            if not text:
                continue  # adjacent markers — a breakpoint already sits here
            blocks.append({
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral", "ttl": ttl},
            })
        tail = "\n\n".join(s.strip() for s in tail_segments if s.strip())
        if tail:
            blocks.append({"type": "text", "text": tail})
        return blocks

    def _with_cache_breakpoint(self, message: dict, ttl: str) -> dict | None:
        """Return a copy of `message` with a cache breakpoint on its last content block.

        The breakpoint takes history_cache_ttl (5m by default, see DevelopmentConfig), the
        system blocks static_cache_ttl (1h): Anthropic accepts a shorter TTL after a longer one.
        The 1h tier on the history was meant for the hourly autonomous check and a chat resumed
        within the hour, which then read the window instead of re-writing it (a 1h write costs
        2x, a read 0.025x) — but the history is re-written on nearly every request anyway (each
        round of a turn, and after every window trim), so the 2x write rarely got its read; see
        CLAUDE_HISTORY_CACHE_TTL for the measurement.

        Returns None when the breakpoint would land on an empty text block: the
        Anthropic API rejects cache_control on empty text ("cache_control cannot be
        set for empty text blocks"), and a blank turn (e.g. an interrupted reply
        that got stored empty) must not crash the whole request. Skipping the
        breakpoint just forgoes caching for that one turn.
        """
        content = message['content']
        if isinstance(content, list):
            if not content:
                return None
            blocks = [block.copy() if isinstance(block, dict) else block for block in content]
            last = blocks[-1]
            if isinstance(last, dict) and last.get("type", "text") == "text" and not (last.get("text") or "").strip():
                return None
            blocks[-1] = {**last, "cache_control": {"type": "ephemeral", "ttl": ttl}}
            return {'role': message['role'], 'content': blocks}
        if not content.strip():
            return None
        return {
            'role': message['role'],
            'content': [{"type": "text", "text": content, "cache_control": {"type": "ephemeral", "ttl": ttl}}],
        }

    def _split_test_suite_tail(self, anthropic_messages: list[dict], system) -> None:
        """See TEST_SUITE_CACHED_TAIL: the user turn starting with it becomes two text blocks —
        the toolkit result with a breakpoint, the rest (the test's query, or on an older model
        the query and the trailing directives) after it. The same bytes, one more breakpoint;
        skipped when the request's four are spoken for (three cached system blocks plus the
        history one)."""
        cached_system = sum(1 for block in system if "cache_control" in block) if isinstance(system, list) else 0
        if cached_system + 2 > 4:
            trace("Claude cache: no breakpoint left for the test suite's toolkit turn")
            return
        tail = self.TEST_SUITE_CACHED_TAIL
        for i, message in enumerate(anthropic_messages):
            content = message.get('content')
            if message['role'] != 'user' or not isinstance(content, str) or not content.startswith(tail):
                continue
            ttl = self.FOLLOW_UP_CACHE_TTL if self.per_request_prefix else self.history_cache_ttl
            blocks = [{"type": "text", "text": tail, "cache_control": {"type": "ephemeral", "ttl": ttl}}]
            rest = content[len(tail):]
            if rest.strip():
                blocks.append({"type": "text", "text": rest})
            anthropic_messages[i] = {'role': 'user', 'content': blocks}
            trace(f"Claude cache: test suite — breakpoint on the toolkit turn (message {i}), "
                  f"{len(rest)} chars of query after it")
            return
        trace("Claude cache: test suite — no turn starts with the toolkit result, nothing split")

    def _build_api_params(self, messages: list[dict], thinking: Thinking | bool, stream: bool, volatile_tail: int = 0,
                          follow_up: bool = False) -> dict:
        """Build common API call parameters."""
        full = Thinking.of(thinking).full   # MORE is an ordinary request here: effort is high regardless
        system_message, anthropic_messages = self._convert_messages(messages)

        if system_message and self.prompt_cache:
            system = self._build_cached_system(system_message)
        elif system_message:
            system = system_message.replace(self.CACHE_BREAKPOINT_MARKER, "")
        else:
            system = None

        index = self._message_cache_index(volatile_tail, follow_up)
        if follow_up:
            trace(f"Claude cache: follow-up round — the tail is cached for {self.FOLLOW_UP_CACHE_TTL}, "
                  "the next round reads it")
        elif volatile_tail:
            trace(f"Claude cache: {volatile_tail} trailing message(s) no later request repeats — "
                  f"history breakpoint on message {index}")
        if index == -1 and self._carries_image(anthropic_messages[-1]):
            # The history keeps an image turn as its text alone (RequestProcessor), so the
            # turn as sent never comes back: the breakpoint goes on the message before it.
            trace("Claude cache: the trailing turn carries an image — history breakpoint on the message before it")
            index = -2
        if self.prompt_cache and self.TEST_SUITE_CACHED_TAIL:
            self._split_test_suite_tail(anthropic_messages, system)
        if self.prompt_cache and len(anthropic_messages) >= max(2, -index):
            ttl = self.FOLLOW_UP_CACHE_TTL if follow_up or self.per_request_prefix else self.history_cache_ttl
            cached = self._with_cache_breakpoint(anthropic_messages[index], ttl)
            if cached is not None:
                anthropic_messages[index] = cached
            else:
                trace(f"Claude cache: message {index} is empty — no history breakpoint this request")

        api_params = {
            'model': self.model,
            'messages': anthropic_messages,
            'max_tokens': self.THINKING_MAX_TOKENS if full else self.MAX_TOKENS,
            'stream': stream
        }
        if not stream:
            api_params['timeout'] = self.NON_STREAMING_TIMEOUT

        if system:
            api_params['system'] = system

        if full:
            api_params['thinking'] = {"type": "adaptive"}
        if self.model.startswith(self.EFFORT_MODEL_PREFIXES):
            api_params['output_config'] = {"effort": self.EFFORT}

        return api_params

    @staticmethod
    def _cache_creation_breakdown(usage, cache_created: int) -> tuple[int, int]:
        """Split cache-creation tokens into (5-minute, 1-hour) writes.

        Anthropic reports the per-TTL split under usage.cache_creation. When that
        detail is absent (older API shape, or a mock without it), attribute all
        created tokens to the cheaper 5-minute rate so the two always sum to the
        reported total.
        """
        detail = getattr(usage, 'cache_creation', None)
        if detail is not None:
            c5m = getattr(detail, 'ephemeral_5m_input_tokens', None)
            c1h = getattr(detail, 'ephemeral_1h_input_tokens', None)
            if isinstance(c5m, int) or isinstance(c1h, int):
                return (c5m if isinstance(c5m, int) else 0), (c1h if isinstance(c1h, int) else 0)
        return cache_created, 0

    def _log_cache_usage(self, usage):
        cache_created = getattr(usage, 'cache_creation_input_tokens', 0) or 0
        cache_read = getattr(usage, 'cache_read_input_tokens', 0) or 0
        input_tokens = getattr(usage, 'input_tokens', 0) or 0
        created_5m, created_1h = self._cache_creation_breakdown(usage, cache_created)
        total = input_tokens + cache_created + cache_read
        if total > 0:
            pct = round(cache_read / total * 100)
            trace(f"Claude cache: {cache_read} read, {cache_created} created "
                  f"({created_5m} 5m + {created_1h} 1h), {input_tokens} uncached "
                  f"(total {total}, {pct}% cached)")
            from common.TTFTMonitor import TTFTMonitor
            TTFTMonitor.get_instance().record_cache_pct(pct)

    def _log_output_tokens(self, usage):
        raw_output = getattr(usage, 'output_tokens', None)
        output_tokens = raw_output if isinstance(raw_output, int) else 0
        if output_tokens > 0:
            trace(f"Claude output: {output_tokens} tokens")

    def _create_message(self, api_params: dict):
        if self.speed == "fast":
            return self.client.beta.messages.create(
                **api_params, speed="fast", betas=[self.FAST_MODE_BETA]
            )
        return self.client.messages.create(**api_params)

    OVERLOADED_RETRY_DELAYS = [0.5, 1.5, 3, 3, 3]

    # Fable (claude-fable-5) occasionally completes a turn with only 2-6 output tokens
    # and no text block at all — observed repeatedly in prod (e.g. 2026-06-11 10:29).
    # Nothing was streamed downstream in those cases, so re-issuing the identical
    # request is safe and has so far always produced a normal reply.
    EMPTY_REPLY_RETRIES = 2

    @staticmethod
    def _raise_if_workspace_required(e: APIStatusError):
        """The one 400 whose fix is in config.yaml, not in the request: reworded into what to do."""
        if workspace_required(e):
            info(f"Claude refused the request: the key names no workspace and {WORKSPACE_CONFIG_KEY} is not set")
            raise WorkspaceRequiredError(WORKSPACE_REQUIRED) from e

    @staticmethod
    def _is_retryable(e: APIStatusError) -> bool:
        if e.status_code in (500, 529):
            return True
        return (hasattr(e, 'body') and isinstance(e.body, dict)
                and isinstance(e.body.get('error'), dict)
                and e.body['error'].get('type') == 'overloaded_error')

    def _stream_once(self, api_params: dict) -> Generator[str, None, None]:
        """One streaming request: yield text deltas, log cache/output usage.

        When the stream completes without a single text delta (the Fable
        empty-reply misfire), log stop_reason and the block types we did get so
        the failure mode is diagnosable from floxy.out.
        """
        first_token_seen = False
        stop_reason = None
        block_types = []
        self._last_stop_reason = None
        stream = self._create_message(api_params)
        try:
            for event in stream:
                if event.type == 'message_start' and hasattr(event.message, 'usage'):
                    self._log_cache_usage(event.message.usage)
                elif event.type == 'content_block_start':
                    block_types.append(getattr(event.content_block, 'type', '?'))
                elif event.type == 'content_block_delta' and event.delta.type == 'text_delta':
                    first_token_seen = True
                    yield event.delta.text
                elif event.type == 'message_delta':
                    stop_reason = getattr(event.delta, 'stop_reason', None) or stop_reason
                    if hasattr(event, 'usage'):
                        self._log_output_tokens(event.usage)
        finally:
            # A consumer that stops reading (the reply cut after the command cap) closes this
            # generator; closing the SDK stream with it ends the request, and the generation.
            close = getattr(stream, "close", None)
            if close is not None:
                close()
        self._last_stop_reason = stop_reason
        if not first_token_seen:
            info(f"Claude stream ended with no text (stop_reason={stop_reason}, blocks={block_types or 'none'})")

    def _stream_with_status_retry(self, api_params: dict) -> Generator[str, None, None]:
        """Stream with retries on retryable API errors (500/529/overloaded) and on
        transport drops (long-lived SSE streams occasionally get cut by upstream edges).

        Transport drops are only retried while nothing has been yielded downstream —
        after that a re-issued request would duplicate the stream, so the error
        propagates for DialogManager to handle (it knows whether the yielded deltas
        were user-facing or just thinking text)."""
        max_attempts = len(self.OVERLOADED_RETRY_DELAYS) + 1
        for attempt, delay in enumerate(self.OVERLOADED_RETRY_DELAYS, start=1):
            yielded = 0
            try:
                for chunk in self._stream_once(api_params):
                    yielded += 1
                    yield chunk
                return
            except APIStatusError as e:
                if not self._is_retryable(e):
                    raise
                info(f"Claude API error (attempt {attempt}/{max_attempts}, status={e.status_code}), retrying in {delay}s...")
                time.sleep(delay)
            except (httpx.TransportError, APIConnectionError) as e:
                if yielded:
                    raise
                info(f"Claude stream dropped before any output ({type(e).__name__}, "
                     f"attempt {attempt}/{max_attempts}), retrying in {delay}s...")
                time.sleep(delay)
        # Final attempt — let exceptions propagate
        yield from self._stream_once(api_params)

    def output_cap(self) -> int | None:
        return self.MAX_TOKENS

    def cancel(self):
        self._cancelled = True

    def create_streaming_completion(self, messages: list[dict], thinking: Thinking | bool = Thinking.NORMAL,
                                    volatile_tail: int = 0, follow_up: bool = False) -> Generator[str, None, None]:
        self._ensure_initialized()
        self._cancelled = False
        self._last_stop_reason = None

        api_params = self._build_api_params(messages, thinking, stream=True, volatile_tail=volatile_tail,
                                            follow_up=follow_up)
        for attempt in range(self.EMPTY_REPLY_RETRIES + 1):
            produced_text = False
            try:
                for chunk in self._stream_with_status_retry(api_params):
                    if chunk.strip():
                        produced_text = True
                    yield chunk
            except APIStatusError as e:
                self._raise_if_workspace_required(e)
                raise
            if self._last_stop_reason == 'max_tokens':
                # The reply is cut, not empty — a resend of the same request thinks itself
                # dry the same way (three identical, fully cached attempts, issue #108).
                # The caller reads the stop reason and tells the model what happened.
                info(f"Claude hit max_tokens={api_params['max_tokens']} - the reply is cut short"
                     + ("" if produced_text else " while still thinking, no text to pass on") + ", not retrying.")
                return
            if produced_text:
                return
            # The safety classifier is deterministic on identical input: re-sending only
            # adds another refused request to the account, so the first refusal is final.
            if self._last_stop_reason == 'refusal':
                info("Claude refused the request (stop_reason=refusal) - not retrying, raising LLMRefusalError.")
                raise LLMRefusalError(f"{self.model} refused the request (stop_reason=refusal)")
            if self._cancelled:
                info("Claude replied with empty message, but the request was killed - not re-sending.")
                return
            if attempt < self.EMPTY_REPLY_RETRIES:
                info(f"Claude replied with empty message (attempt {attempt + 1}/{self.EMPTY_REPLY_RETRIES + 1}), "
                     "re-sending the request...")
        info("Claude replied with empty message on every retry - passing the empty reply through.")

    def _complete_once(self, api_params: dict) -> str:
        max_attempts = len(self.OVERLOADED_RETRY_DELAYS) + 1
        response = None
        for attempt, delay in enumerate(self.OVERLOADED_RETRY_DELAYS, start=1):
            try:
                response = self._create_message(api_params)
                break
            except APIStatusError as e:
                if not self._is_retryable(e):
                    raise
                info(f"Claude API error (attempt {attempt}/{max_attempts}, status={e.status_code}), retrying in {delay}s...")
                time.sleep(delay)
            except (httpx.TransportError, APIConnectionError) as e:
                info(f"Claude request failed on transport ({type(e).__name__}, "
                     f"attempt {attempt}/{max_attempts}), retrying in {delay}s...")
                time.sleep(delay)
        if response is None:
            response = self._create_message(api_params)
        self._log_cache_usage(response.usage)
        self._log_output_tokens(response.usage)
        self._last_stop_reason = getattr(response, 'stop_reason', None)
        if self._last_stop_reason == 'max_tokens':
            info(f"Claude hit max_tokens={api_params['max_tokens']} - the reply is cut short")

        text = ''.join(block.text for block in response.content if block.type == 'text')
        if not text.strip():
            block_types = [getattr(block, 'type', '?') for block in response.content]
            info(f"Claude returned no text (stop_reason={getattr(response, 'stop_reason', None)}, "
                 f"blocks={block_types or 'none'})")
        return text

    def create_completion(self, messages: list[dict], thinking: Thinking | bool = Thinking.NORMAL, volatile_tail: int = 0,
                          follow_up: bool = False) -> str:
        self._ensure_initialized()

        api_params = self._build_api_params(messages, thinking, stream=False, volatile_tail=volatile_tail,
                                            follow_up=follow_up)
        text = ""
        for attempt in range(self.EMPTY_REPLY_RETRIES + 1):
            try:
                text = self._complete_once(api_params)
            except APIStatusError as e:
                self._raise_if_workspace_required(e)
                raise
            if text.strip() or self._last_stop_reason == 'max_tokens':
                return text
            # Same rule as the streaming path: the classifier is deterministic, so the
            # first refusal is final — re-sending only adds refused requests.
            if self._last_stop_reason == 'refusal':
                info("Claude refused the request (stop_reason=refusal) - not retrying, raising LLMRefusalError.")
                raise LLMRefusalError(f"{self.model} refused the request (stop_reason=refusal)")
            if attempt < self.EMPTY_REPLY_RETRIES:
                info(f"Claude replied with empty message (attempt {attempt + 1}/{self.EMPTY_REPLY_RETRIES + 1}), "
                     f"re-sending the request...")
        info("Claude replied with empty message on every retry - returning the empty reply.")
        return text

    # A batch is polled, not streamed: most end within the hour, so half a minute between
    # looks costs nothing. After a cancel the batch still has to wind down before its result
    # (the reply, if it finished in the meantime) can be read.
    BATCH_POLL_SECONDS = 30
    BATCH_CANCEL_WAIT_SECONDS = 300

    def create_batched_completion(self, messages: list[dict], thinking: Thinking | bool = Thinking.NORMAL,
                                  max_wait_seconds: float = 3600) -> str | None:
        """One request through the Message Batches API — every token at half price, the answer
        whenever Anthropic gets to it (usually well within the hour, 24 hours at most). A batch
        that has not ended after `max_wait_seconds` is cancelled; a reply it produced while the
        cancel was on its way is still used, so the caller never pays for the same answer
        twice. None when the batch gave no reply. Fast mode is not available on batches."""
        self._ensure_initialized()
        if self.speed == "fast":
            info("Claude batch: fast mode is not available on the Batches API - no batch")
            return None

        api_params = self._build_api_params(messages, thinking, stream=False)
        del api_params['stream'], api_params['timeout']
        self._last_stop_reason = None
        t_start = time.time()
        batch = self.client.messages.batches.create(requests=[{'custom_id': 'request', 'params': api_params}])
        info(f"Claude batch {batch.id} submitted ({self.model}), waiting up to {max_wait_seconds / 60:.0f} min...")

        if not self._wait_for_batch(batch.id, t_start + max_wait_seconds):
            info(f"Claude batch {batch.id} has not ended after {(time.time() - t_start) / 60:.0f} min - cancelling it")
            self.client.messages.batches.cancel(batch.id)
            if not self._wait_for_batch(batch.id, time.time() + self.BATCH_CANCEL_WAIT_SECONDS):
                info(f"Claude batch {batch.id} is still winding down - giving up on it")
                return None

        result = next(iter(self.client.messages.batches.results(batch.id)), None)
        outcome = getattr(getattr(result, 'result', None), 'type', None)
        if outcome != 'succeeded':
            info(f"Claude batch {batch.id} ended without a reply ({outcome or 'no result'})")
            return None
        response = result.result.message
        info(f"Claude batch {batch.id} answered after {(time.time() - t_start) / 60:.1f} min")
        self._log_cache_usage(response.usage)
        self._log_output_tokens(response.usage)
        self._last_stop_reason = getattr(response, 'stop_reason', None)
        if self._last_stop_reason == 'refusal':
            info("Claude refused the batched request (stop_reason=refusal) - raising LLMRefusalError.")
            raise LLMRefusalError(f"{self.model} refused the request (stop_reason=refusal)")
        if self._last_stop_reason == 'max_tokens':
            info(f"Claude hit max_tokens={api_params['max_tokens']} - the reply is cut short")
        text = ''.join(block.text for block in response.content if block.type == 'text')
        if not text.strip():
            info(f"Claude batch {batch.id} returned no text (stop_reason={self._last_stop_reason})")
            return None
        return text

    def _wait_for_batch(self, batch_id: str, deadline: float) -> bool:
        """Poll until the batch has ended (True) or the deadline passes (False). A poll that
        fails on the network or on an overloaded API is just a missed look."""
        while True:
            try:
                status = self.client.messages.batches.retrieve(batch_id).processing_status
                if status == 'ended':
                    return True
                trace(f"Claude batch {batch_id}: {status}")
            except (httpx.TransportError, APIConnectionError) as e:
                info(f"Claude batch {batch_id}: poll failed on transport ({type(e).__name__}), looking again later")
            except APIStatusError as e:
                if not self._is_retryable(e):
                    raise
                info(f"Claude batch {batch_id}: poll failed (status={e.status_code}), looking again later")
            if time.time() >= deadline:
                return False
            time.sleep(self.BATCH_POLL_SECONDS)

    def create_completion_with_images(self, messages: list[dict], images: list, background: bool = False) -> str:
        self._ensure_initialized()

        system_message, anthropic_messages = self._convert_messages(messages)

        content_blocks = []
        for img in images:
            if isinstance(img, (bytes, bytearray)):
                import base64
                data = base64.b64encode(img).decode('utf-8')
                media_type = "image/png" if img[:4] == b'\x89PNG' else "image/jpeg"
            else:
                data = img
                media_type = "image/jpeg"
            content_blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data}
            })

        if anthropic_messages and anthropic_messages[-1]['role'] == 'user':
            last_text = anthropic_messages[-1]['content']
            content_blocks.append({"type": "text", "text": last_text})
            anthropic_messages[-1] = {'role': 'user', 'content': content_blocks}

        api_params = {
            'model': self.model,
            'messages': anthropic_messages,
            'max_tokens': 4096,
        }
        if system_message and self.prompt_cache:
            # The system prompt is cached like a chat request's: with breakpoint markers
            # (a camera question from the chat) block by block, without them (the photo
            # indexer's, identical from photo to photo — the memory bank and the
            # instructions, several thousand tokens against one image) as one 5-minute
            # block, kept warm by the backfill's back-to-back requests. The image and the
            # per-photo lines stay in the user turn, after the cached prefix.
            api_params['system'] = self._build_cached_system(system_message)
        elif system_message:
            api_params['system'] = self.strip_cache_markers(system_message)

        response = self.client.messages.create(**api_params)
        usage = getattr(response, 'usage', None)
        if usage is not None:
            self._log_cache_usage(usage)
            self._log_output_tokens(usage)
        text = ''.join(block.text for block in response.content if block.type == 'text')
        if not text.strip():
            block_types = [getattr(block, 'type', '?') for block in response.content]
            info(f"Claude (image call) returned no text (stop_reason={getattr(response, 'stop_reason', None)}, "
                 f"blocks={block_types or 'none'})")
        return text
