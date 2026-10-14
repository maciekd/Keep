import json
import time
from datetime import datetime
from typing import Callable

from assistant.ChannelProfile import WINDOW_TEXT
from common.logging import info, trace
from common.FloxyDatabase import FloxyDatabase, ROLE_COMMAND as DB_ROLE_COMMAND
from ..tools.MessageFormatter import MessageFormatter
from .HistoryCompressor import HistoryCompressor

# How a model without thinking of its own reasons before it answers (the tag is stripped before
# TTS). PromptManager.get_messages merges it into the trailing user turn; a model that handles
# thinking internally (AbstractLLM.handles_thinking_internally) must never receive it.
THINKING_TAG_INSTRUCTION = ("If the query is complex enough to warrant it, start your reply by explaining your "
                            "thinking inside a <thinking> tag. Skip the tag for simple queries. If you do think, "
                            "ALL reasoning must stay inside the <thinking> tag. NO other tags are allowed.")


class MessageHistory:
    """Owns the conversation history (user/assistant turns + system/context events).

    History is no longer baked into the prompt text — it is rendered as standard
    LLM message items via get_history_messages() and appended after the system
    prompt at request time. Feeds (iMessage, Telegram, Photos, Indigo, Home Assistant) pour their rows in
    through add_context_messages. Subscribers (the TUI's HistoryFeed) are called after
    every change, on the thread that made it.

    One history, two windows (ChannelProfile.window): a row a preload put in for one channel
    is stamped with that window in its `source` (`preload:voice`) and rendered for that
    window only, and a carried-section delta note is rendered where the toolkit's reference
    is — so a chat request after a spoken one never carries voice's toolkits (`visible`).
    Everything else of the window — the trim, its cutoff, the eviction stub — is shared and
    counts the hidden rows too; the cutoff is a timestamp, so each window's head stays as
    stable as before.
    """

    MAX_RENDER_CHARS = 2000
    # A command's result is stored as its own `command` row after the assistant round that
    # ran it, so the next turn still sees what list_integrations / config_help / a search
    # returned instead of only the assistant's reaction to it. Stored capped at this size
    # (a web page read or a log dump is far bigger) and rendered whole — MAX_RENDER_CHARS
    # would cut a list_integrations table in half. Once a later row supersedes it (a newer
    # browse snapshot) HistoryCompressor collapses it at render time, before the trim.
    ROLE_COMMAND = DB_ROLE_COMMAND
    COMMAND_RESULT_CHARS = 6000
    # The clock note a byte-prefix request put in front of its query (PromptManager.get_messages),
    # persisted as its own verbatim row so the next request renders the same bytes.
    ROLE_NOTE = "note"
    # Hysteresis watermarks for the LLM history window: the rendered window grows
    # to HIGH, is then cut back to LOW, and the head stays frozen until HIGH is
    # hit again. A plain per-render budget evicted the oldest entry on nearly
    # every turn, shifting the window head and invalidating the message-level KV
    # cache from the first history turn onward — the whole ~6K-token history was
    # re-written on every request (see _hysteresis_trim). Each trim still costs a
    # rewrite of the window at the 1h cache-write rate (2x), while carrying a bigger
    # window costs reads at 0.025x: at 25KB prod trimmed 19 times on 2026-10-09,
    # $2.50 of rewrites, and 35KB halves that for a few cents of extra reads.
    TRIM_HIGH_BYTES = 35 * 1024
    TRIM_LOW_BYTES = 15 * 1024
    # A trim at HIGH waits while the newest row it would drop is younger than
    # TRIM_FRESH_SECONDS: a burst (a checkout, a browsing session) fills the window with
    # the rows the task is using, and cutting at HIGH evicted the browser toolkit 75 s
    # after it was loaded and the user's instructions a minute after they were given
    # (2026-10-05). A quiet afternoon's stale rows still go at HIGH. At TRIM_CEILING_BYTES
    # the trim happens whatever the rows' age — the ceiling is what the window costs in any
    # busy period, feed bursts included, so it is a size worth paying always.
    TRIM_CEILING_BYTES = 50 * 1024
    TRIM_FRESH_SECONDS = 15 * 60
    # A trim that evicts toolkit reference rows leaves one transient `system` stub naming the
    # evicted toolkits, at the newest evicted row's timestamp and exempt from the cutoff, so
    # the LLM knows the commands it saw there are gone — without it the model keeps using
    # the toolkit from memory and guesses at names (a live run emitted browse_go for
    # browse_open). The prompt's toolkit section explains the note; the stub is only the
    # names, since a voice day evicts a dozen at once (12 one-line stubs, 2.2KB, 2026-10-12).
    # A name leaves the stub when its toolkit is loaded again; the stub goes at the next
    # trim, with the day's wipe, or when its last name leaves.
    SOURCE_TOOLKIT_EVICTED = "toolkit_evicted"
    # The `source` of a `command` row a preload put in for one window: this prefix and the
    # window's name (preload_source / preload_window). A command row the LLM loaded itself
    # has no source and is every window's.
    SOURCE_PRELOAD_PREFIX = "preload:"
    SOURCE_PROMPT_DELTA = "prompt_delta"

    def __init__(self):
        self.history = []
        # Transient entries are in-memory only — they ride alongside persisted history
        # for rendering but are never written to floxy.db. Used by PromptManager to inject
        # "system prompt diff" notes between full prompt rebuilds, so the cached system
        # prompt bytes can stay stable while the LLM still sees up-to-date section data.
        self.transient_entries: list[dict] = []
        self._subscribers: list[Callable[[], None]] = []
        self.last_update_time = 0.0   # shown in the prompt section stats
        self.wipes = 0                # bumped by wipe_today so subscribers know rows vanished
        # Generous entry cap — the byte watermarks below are the real limit. Must
        # stay well above what TRIM_HIGH_BYTES can hold: if this cap binds first,
        # the DB window slides one entry per exchange and the head shifts every
        # turn again, silently defeating the hysteresis.
        self.max_entries = 500
        self.max_render_chars = self.MAX_RENDER_CHARS
        self.trim_high_bytes = self.TRIM_HIGH_BYTES
        self.trim_low_bytes = self.TRIM_LOW_BYTES
        self.trim_ceiling_bytes = self.TRIM_CEILING_BYTES
        self.trim_fresh_seconds = self.TRIM_FRESH_SECONDS
        # Timestamp of the oldest entry in the LLM window; entries older than this
        # are hidden from get_history_messages even if they'd fit. Only advances
        # when the window exceeds trim_high_bytes (see _hysteresis_trim).
        self._trim_cutoff = 0.0
        self._deferred_trim_at = 0    # window size whose trim was last deferred, so the deferral logs once
        # Section keys whose prompt-delta notes a trim evicted. The notes are the only
        # thing telling the LLM the section moved on from the system prompt's baseline, so
        # PromptManager drains this set before every send and puts one fresh note back.
        self.dropped_delta_sections: set[str] = set()
        self._compressor = HistoryCompressor()
        self._load_history()

    def subscribe(self, callback: Callable[[], None]):
        """Call `callback` (no arguments) after every change to the merged entries."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[], None]):
        if callback in self._subscribers:
            self._subscribers.remove(callback)
        else:
            trace("MessageHistory: unsubscribe of a callback that was not subscribed")

    def _notify_changed(self):
        self.last_update_time = time.time()
        for callback in list(self._subscribers):
            try:
                callback()
            except Exception as e:
                info(f"MessageHistory: subscriber failed: {e}")

    def _load_history(self):
        """Load recent history from database."""
        try:
            self.history = FloxyDatabase.get_instance().get_recent_history(self.max_entries)
        except Exception:
            self.history = []

    @staticmethod
    def _save_history_entry(role, content, timestamp, source=None, location=None):
        """Persist a single history entry to the database."""
        try:
            FloxyDatabase.get_instance().add_history_entry(role, content, timestamp, source, location)
        except Exception as e:
            trace(f"Failed to save history entry: {e}")

    def wipe_today(self) -> int:
        """Delete today's persisted history (the developer's "clear" on the Prompt tab) and
        drop the transient notes. Returns the number of rows deleted. The LLM window starts
        over from what remains; `wipes` tells subscribers the entries they hold are gone,
        not merely that new ones landed."""
        deleted = FloxyDatabase.get_instance().delete_todays_history()
        self.transient_entries = []
        self._trim_cutoff = 0.0
        self._load_history()
        self.wipes += 1
        info(f"MessageHistory: wiped today's history ({deleted} rows), {len(self.history)} older entries remain")
        self._notify_changed()
        return deleted

    def get_merged_entries(self, max_entries: int | None = None) -> list[dict]:
        """Get history entries — persisted DB rows merged with in-memory transient entries, sorted by timestamp."""
        if not self.history and not self.transient_entries:
            return []

        if self.transient_entries:
            merged = sorted(
                list(self.history) + list(self.transient_entries),
                key=lambda e: e["timestamp"],
            )
        else:
            merged = list(self.history)

        limit = max_entries if max_entries is not None else self.max_entries
        if len(merged) > limit:
            return merged[-limit:]
        return merged

    def add_transient_entry(self, role: str, content: str, timestamp: float,
                            source: str | None = None, location: str | None = None,
                            section: str | None = None):
        """Add an in-memory-only history entry. NOT persisted to floxy.db.

        Used by PromptManager to inject section-diff notes (see
        PromptManager._emit_transient_section_update). `section` tags the entry with the
        section it describes. Once in the window a note stays until the trim drops it:
        removing or rewording a row changes the window's bytes from that row on, and the
        prefix cache reads the window only while every byte before the breakpoint repeats.
        """
        self.transient_entries.append({
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "source": source,
            "location": location,
            "section": section,
        })
        trace(f"MessageHistory: transient entry added ({role}, {len(content)} chars, source={source})")
        self._notify_changed()

    def _prune_transient_entries_before_cutoff(self):
        """Forget the transient notes the trim left behind the cutoff. They can never render
        again (the cutoff only advances), so this changes no byte of the window — it is
        memory hygiene, the one way a note ever leaves the list short of a wipe. The
        eviction stub stays: it stands at the evicted row's time, behind the cutoff by design."""
        kept = [e for e in self.transient_entries
                if e["timestamp"] >= self._trim_cutoff or self._is_eviction_stub(e)]
        pruned = len(self.transient_entries) - len(kept)
        if pruned:
            self.transient_entries = kept
            trace(f"MessageHistory: {pruned} transient note(s) behind the trim cutoff forgotten")

    @classmethod
    def preload_source(cls, window: str) -> str:
        """The `source` a preloaded toolkit row carries for `window`."""
        return cls.SOURCE_PRELOAD_PREFIX + window

    @classmethod
    def preload_window(cls, entry: dict) -> str | None:
        """The window a row was preloaded for, None for a row every window renders."""
        source = entry.get("source") or ""
        return source[len(cls.SOURCE_PRELOAD_PREFIX):] if source.startswith(cls.SOURCE_PRELOAD_PREFIX) else None

    @classmethod
    def visible(cls, rows: list[tuple[dict, str, str]], window: str) -> list[tuple[dict, str, str]]:
        """The rendered rows `window` may see: without the rows preloaded for another window,
        and without the delta notes of a carried section whose toolkit reference the window
        does not hold (the note diffs against data the LLM was never sent). A note always
        comes after the reference it belongs to, so one forward pass decides."""
        from assistant.commands.Toolkits import CARRIED_SECTIONS
        kept, held, hidden = [], set(), 0
        for row in rows:
            entry = row[0]
            preloaded_for = cls.preload_window(entry)
            if preloaded_for is not None and preloaded_for != window:
                hidden += 1
                continue
            if entry.get("source") == cls.SOURCE_PROMPT_DELTA:
                toolkit = CARRIED_SECTIONS.get(entry.get("section"))
                if toolkit is not None and toolkit not in held:
                    hidden += 1
                    continue
            held.update(cls._toolkit_names(entry))
            kept.append(row)
        if hidden:
            trace(f"MessageHistory: {hidden} row(s) of the window are not the {window} window's — left out")
        return kept

    def get_history_messages(self, max_bytes: int | None = None, window: str = WINDOW_TEXT) -> list[dict]:
        """Render history as standard LLM message items (role/content dicts) for `window`
        (ChannelProfile.window, see `visible`).

        Each entry maps to a user/assistant turn (system/context events ride on
        the user side). Named-user entries are formatted byte-identically to the
        live query (MessageFormatter.format_message) so a saved exchange matches
        the bytes previously sent — keeping the KV/prompt cache warm turn-to-turn.
        Consecutive same-role turns are coalesced (required by e.g. Gemini) and
        any leading non-user turns are dropped so the list always starts with a
        user message.

        Superseded rows are compressed first (HistoryCompressor), then the window is
        trimmed with hysteresis (see _hysteresis_trim) so its head stays byte-stable
        across turns and the message-level KV cache keeps hitting. Passing an explicit
        max_bytes renders a one-shot view trimmed to that budget instead, without
        touching the sticky window.
        """
        if max_bytes is not None:
            return self.render_messages(self.get_merged_entries(), max_bytes, self.max_render_chars)
        return self._finalize_messages([(role, content) for _, role, content in self._window(window)])

    def _window(self, window: str) -> list[tuple[dict, str, str]]:
        """The live history window as (entry, role, content) rows: every entry rendered,
        superseded rows compressed, the hysteresis trim applied, then `window`'s rows of it
        (`visible`). get_history_messages sends it and render_for_display shows it, so the
        Prompt tab never drifts from the request."""
        rendered = self._compressor.compress(self._render_entries(self.get_merged_entries(), self.max_render_chars))
        return self.visible(self._hysteresis_trim(rendered), window)

    def trim_window(self):
        """Advance the sticky trim cutoff the way the next get_history_messages() would, so a
        caller deciding on the window's contents beforehand (ToolkitPreloader — is the toolkit
        still in it?) sees the window the request is about to send, not the pre-trim one."""
        before = self._trim_cutoff
        self.get_history_messages()
        if self._trim_cutoff != before:
            trace(f"MessageHistory: trim_window advanced the cutoff to "
                  f"{datetime.fromtimestamp(self._trim_cutoff).strftime('%H:%M:%S')}")

    def _hysteresis_trim(self, rendered: list[tuple[dict, str, str]]) -> list[tuple[dict, str, str]]:
        """Trim the LLM history window with hysteresis: grow to trim_high_bytes,
        then cut back to trim_low_bytes and freeze the head there (_trim_cutoff).

        The cutoff is sticky so the rendered prefix — and with it the message-level
        KV cache — survives turn-to-turn: the head only moves once per
        ~(high - low) bytes of new conversation instead of on every exchange.

        Between high and trim_ceiling_bytes the cut waits while the newest row it would
        drop is younger than trim_fresh_seconds (see TRIM_CEILING_BYTES); the age is
        checked after snapping, since the snap can move the cut and with it that row.
        """
        stub = next((entry for entry, _, _ in rendered if self._is_eviction_stub(entry)), None)
        rendered = [item for item in rendered if not self._is_eviction_stub(item[0])]
        kept = [item for item in rendered if item[0]["timestamp"] >= self._trim_cutoff]
        total = sum(len(content.encode("utf-8")) for _, _, content in kept)
        if total <= self.trim_high_bytes:
            return self._with_eviction_stub(stub, kept, dropped=[])

        # Over the high watermark: walk newest→oldest to find the low-watermark cut.
        keep_from = len(kept)
        kept_bytes = 0
        for i in range(len(kept) - 1, -1, -1):
            size = len(kept[i][2].encode("utf-8"))
            if kept_bytes + size > self.trim_low_bytes:
                break
            kept_bytes += size
            keep_from = i
        keep_from = self._snap_to_boundary(kept, keep_from, self.trim_high_bytes)
        if keep_from == 0:
            trace("MessageHistory: window over high watermark but nothing trimmable — leaving it")
            return self._with_eviction_stub(stub, kept, dropped=[])
        newest_dropped_age = time.time() - kept[keep_from - 1][0]["timestamp"]
        fresh = newest_dropped_age < self.trim_fresh_seconds
        if fresh and total <= self.trim_ceiling_bytes:
            if total != self._deferred_trim_at:   # the TUI renders the window many times a second
                self._deferred_trim_at = total
                trace(f"MessageHistory: window at {total / 1024:.1f}KB (high {self.trim_high_bytes // 1024}KB) but "
                      f"the newest row a trim would drop is {newest_dropped_age / 60:.1f}m old — waiting for it to "
                      f"age past {self.trim_fresh_seconds // 60}m or the window to hit the "
                      f"{self.trim_ceiling_bytes // 1024}KB ceiling")
            return self._with_eviction_stub(stub, kept, dropped=[])
        kept_bytes = sum(len(content.encode("utf-8")) for _, _, content in kept[keep_from:])

        dropped = kept[:keep_from]
        kept = kept[keep_from:]
        self._trim_cutoff = kept[0][0]["timestamp"]
        self._prune_transient_entries_before_cutoff()
        why = (f"ceiling {self.trim_ceiling_bytes // 1024}KB, newest dropped row only {newest_dropped_age / 60:.1f}m old"
               if fresh else f"high {self.trim_high_bytes // 1024}KB")
        info(f"MessageHistory: window hit {total / 1024:.1f}KB "
             f"({why}) — trimmed {len(dropped)} entries, "
             f"kept {len(kept)} ({kept_bytes / 1024:.1f}KB), new cutoff "
             f"{datetime.fromtimestamp(self._trim_cutoff).strftime('%H:%M:%S')}")
        dropped_notes = [entry for entry, _, _ in dropped if entry.get("source") == self.SOURCE_PROMPT_DELTA]
        if dropped_notes:
            sections = {entry.get("section") for entry in dropped_notes} - {None}
            self.dropped_delta_sections |= sections
            info(f"MessageHistory: trim dropped {len(dropped_notes)} prompt-delta note(s) of "
                 f"{sorted(sections)} — PromptManager restates those sections before the next send")
        return self._with_eviction_stub(stub, kept, dropped)

    # --- toolkit-eviction stubs (see SOURCE_TOOLKIT_EVICTED) ------------------------------

    @classmethod
    def _is_eviction_stub(cls, entry: dict) -> bool:
        return entry.get("source") == cls.SOURCE_TOOLKIT_EVICTED

    @staticmethod
    def _toolkit_names(entry: dict) -> list[str]:
        """The toolkit names a `command` row's reference text carries (none for other rows)."""
        from assistant.commands.Toolkits import loaded_name
        if entry["role"] != MessageHistory.ROLE_COMMAND:
            return []
        return [name for name in (loaded_name(line) for line in entry["content"].split("\n")) if name]

    @staticmethod
    def _stub_text(names: list[str]) -> str:
        return (f"Toolkit references trimmed from the window: {', '.join(names)}. A toolkit loaded "
                f"again below is current; load the others again before using them.")

    def _with_eviction_stub(self, stub: dict | None, kept: list[tuple[dict, str, str]],
                            dropped: list[tuple[dict, str, str]]) -> list[tuple[dict, str, str]]:
        """The window with its one toolkit-eviction stub in front. The stub holds still between
        trims: it is the window's first row, and a byte of it changing (a name leaving when its
        toolkit was loaded again, as it used to) is a cache miss on the whole window — so the
        text points at the reload rows below instead. A trim (`dropped` non-empty) retires the
        earlier trim's stub whole — a toolkit not reloaded within one window's worth of
        conversation is not in use — and leaves a fresh one naming every toolkit row it evicted."""
        in_window = {name for entry, _, _ in kept for name in self._toolkit_names(entry)}
        names = [] if dropped or stub is None else list(stub["toolkits"])

        evicted: dict[str, float] = {}
        for entry, _, _ in dropped:
            for name in self._toolkit_names(entry):
                if name not in in_window:
                    evicted[name] = max(evicted.get(name, 0.0), entry["timestamp"])
        if evicted:
            info(f"MessageHistory: trim evicted the toolkit reference(s) of {', '.join(sorted(evicted))} "
                 f"— one stub names them at the newest row's place")
            names = sorted(set(names) | set(evicted))
        timestamp = max(evicted.values()) if evicted else (stub["timestamp"] if stub else 0.0)

        if stub is not None and (names != stub["toolkits"] or timestamp != stub["timestamp"]):
            self.transient_entries = [e for e in self.transient_entries if e is not stub]
            stub = None
        if names and stub is None:
            stub = {"role": "system", "content": self._stub_text(names), "timestamp": timestamp,
                    "source": self.SOURCE_TOOLKIT_EVICTED, "location": None, "section": None,
                    "toolkits": names}
            self.transient_entries = self.transient_entries + [stub]
        if stub is None:
            return kept
        return [(stub, *self._render_entry(stub, self.max_render_chars))] + kept

    @staticmethod
    def _snap_to_boundary(kept: list[tuple[dict, str, str]], keep_from: int, high_bytes: int) -> int:
        """Move a window-head candidate to a position the sticky cutoff can express.

        Two constraints: the window must start at an exchange boundary (a user-side
        turn — an orphaned assistant reply whose query was trimmed would only get
        popped by _finalize_messages anyway), and it must start at a timestamp
        boundary, since the cutoff is a bare timestamp and the `>=` filter would
        restore any entries tied with the head on the next call — an endless trim
        that logs on every call and never converges. Ties are real: a multi-photo
        iMessage batch stores one row per attachment, all with the same chat.db date.
        Tied entries are pulled back into the window when the result still fits
        under high_bytes; otherwise (the tie runs from the very head, or is too big
        to keep) the cut moves past the tie instead. Never moves past the newest
        entry.
        """
        n = len(kept)
        while keep_from < n and kept[keep_from][1] != "user":
            keep_from += 1
        if keep_from >= n:
            keep_from = n - 1
        head_ts = kept[keep_from][0]["timestamp"]
        tie_start = keep_from
        while tie_start > 0 and kept[tie_start - 1][0]["timestamp"] == head_ts:
            tie_start -= 1
        if keep_from == 0:
            return 0
        if tie_start > 0:
            with_tie = sum(len(content.encode("utf-8")) for _, _, content in kept[tie_start:])
            if with_tie <= high_bytes:
                return tie_start
            trace(f"MessageHistory: tied batch at {datetime.fromtimestamp(head_ts).strftime('%H:%M:%S')} "
                  f"would leave the window at {with_tie / 1024:.1f}KB (> high) — cutting past it")
        while keep_from < n and kept[keep_from][0]["timestamp"] == head_ts:
            keep_from += 1
        while keep_from < n and kept[keep_from][1] != "user":
            keep_from += 1
        if keep_from >= n:
            keep_from = n - 1
        if kept[keep_from][0]["timestamp"] == head_ts:
            return 0  # every entry is tied with the head: nothing can be cut
        return keep_from

    @staticmethod
    def _render_entries(entries: list[dict], max_render_chars: int) -> list[tuple[dict, str, str]]:
        """Every entry rendered to (entry, role, content), blank turns left out."""
        rendered = []
        for entry in entries:
            turn = MessageHistory._render_entry(entry, max_render_chars)
            if turn is not None:
                rendered.append((entry, turn[0], turn[1]))
        return rendered

    @staticmethod
    def _render_entry(entry: dict, max_render_chars: int) -> tuple[str, str] | None:
        """Render one history entry into a (role, content) turn; None for blank turns."""
        role = entry["role"]
        content = entry["content"]
        timestamp = entry["timestamp"]

        if not content or not content.strip():
            # Skip blank turns — e.g. an assistant reply interrupted by a server
            # restart that got stored empty. Sending it back would add nothing and
            # would crash Claude's cache breakpoint (no cache_control on empty text).
            return None

        if role in (MessageHistory.ROLE_COMMAND, MessageHistory.ROLE_NOTE):
            # Byte-identical to what the request sent on the user side (a follow-up round's
            # `result:` turn, a byte-prefix request's clock note), so the persisted history
            # re-renders the same prefix. Capped at store time, never cut here.
            return ("user", content)

        if len(content) > max_render_chars:
            if role == "assistant":
                # Cut from the middle, keeping the tail: assistant turns put the
                # user-visible reply after the (large) <thinking> block, so a head-only
                # cut discards exactly the part worth keeping and leaves a trailing
                # "..." the model misreads as its own unfinished sentence — it then
                # "completes" the message and the user gets a duplicate notification.
                tail = max_render_chars // 4
                head = max_render_chars - tail
                content = content[:head] + " […history truncated…] " + content[-tail:]
            else:
                content = content[:max_render_chars] + "..."

        if role == "assistant":
            # No timestamp prefix on assistant turns: the model imitates any prefix baked
            # into its own prior content and starts emitting "(HH:MM)" in its replies.
            return ("assistant", content)
        if role == "system":
            # System/home-automation events ride on the user side: the message API has no
            # in-timeline system role (only the top-level system prompt), so external input
            # gets the "user" role and the "System (...)" text prefix carries the semantics.
            return ("user", f"System ({MessageFormatter.format_timestamp(timestamp)}): {content}")
        return ("user", MessageFormatter.format_message(
            role, timestamp, entry.get("source"), entry.get("location"), content))

    @staticmethod
    def _finalize_messages(turns: list[tuple[str, str]]) -> list[dict]:
        """Coalesce consecutive same-role turns (required by e.g. Gemini) and drop
        leading non-user turns so the list always starts with a user message."""
        coalesced: list[dict] = []
        for msg_role, msg_content in turns:
            if coalesced and coalesced[-1]["role"] == msg_role:
                # The same separator append_user_turn and DialogManager.stream_llm_response
                # put between a trailing user turn and what they fold onto it, so a rendered
                # history turn is byte-identical to the request that sent its parts.
                coalesced[-1]["content"] += "\n\n" + msg_content
            else:
                coalesced.append({"role": msg_role, "content": msg_content})

        while coalesced and coalesced[0]["role"] != "user":
            coalesced.pop(0)

        return coalesced

    @staticmethod
    def render_messages(entries: list[dict], budget: int, max_render_chars: int) -> list[dict]:
        """One-shot render trimmed to a byte budget (oldest entries dropped first).

        Used by the prompt test suite's fixed fixture (load_fixture_messages) and by
        explicit-budget callers. The live LLM path goes through get_history_messages,
        which trims with hysteresis instead — but both render each turn through
        _render_entry and HistoryCompressor, so equivalent entries produce
        byte-identical turns.
        """
        if not entries:
            return []

        rendered = [(role, content) for _, role, content in
                    HistoryCompressor().compress(MessageHistory._render_entries(entries, max_render_chars))]

        # Keep newest turns that fit within the byte budget (oldest dropped first).
        kept_reversed: list[tuple[str, str]] = []
        total = 0
        for msg_role, msg_content in reversed(rendered):
            addition = len(msg_content.encode("utf-8"))
            if kept_reversed and total + addition > budget:
                break
            total += addition
            kept_reversed.append((msg_role, msg_content))

        return MessageHistory._finalize_messages(list(reversed(kept_reversed)))

    @staticmethod
    def load_fixture_messages(path: str) -> list[dict]:
        """Render a JSON history fixture into LLM message turns (prompt test suite).

        The fixture is a list of entries shaped like FloxyDatabase.get_recent_history()
        rows ({role, content, timestamp, source, location}), with timestamp as a local
        'YYYY-MM-DD HH:MM:SS' string converted to Unix time here. The result is rendered
        once and held fixed, so it sends the exact same turns production sends — without
        the live, mutable history accumulating per-test traffic.
        """
        with open(path, "r") as f:
            raw = json.load(f)

        entries = []
        for entry in raw:
            ts = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S").timestamp()
            entries.append({
                "role": entry["role"],
                "content": entry["content"],
                "timestamp": ts,
                "source": entry.get("source"),
                "location": entry.get("location"),
            })

        return MessageHistory.render_messages(entries, MessageHistory.TRIM_HIGH_BYTES, MessageHistory.MAX_RENDER_CHARS)

    def render_for_display(self, window: str = WINDOW_TEXT) -> str:
        """Render the LLM history window of `window` as markdown for the console PromptTab
        viewer (it toggles between the text and the voice window).

        The rows are the window get_history_messages() sends (_window: rendered,
        compressed, trimmed), so every body is byte for byte what the LLM receives —
        a superseded browse snapshot shows collapsed, a long turn shows cut. Each row
        gets an h3 header (name/time/source; a preloaded row says whose window it is) so
        MarkdownTabView colours it; the entries before the trim cutoff are summarised in
        a single leading note.
        """
        entries = self.get_merged_entries()
        if not entries:
            return "_No conversation history yet._"

        window = self._window(window)
        trimmed = sum(1 for e in entries if e["timestamp"] < self._trim_cutoff and not self._is_eviction_stub(e))
        blocks = []
        if trimmed:
            cutoff = datetime.fromtimestamp(self._trim_cutoff).strftime('%H:%M:%S')
            blocks.append(f"_{trimmed} older entries (before {cutoff}) trimmed from the window — not sent._")
        for entry, _, content in window:
            blocks.append(f"### {self._display_header(entry)}\n{content}")
        return "\n\n".join(blocks)

    @classmethod
    def _display_header(cls, entry: dict) -> str:
        role = entry["role"]
        timestamp = entry["timestamp"]
        if role == "assistant":
            return MessageFormatter.format_assistant_header(timestamp)
        if role == "system":
            return f"System ({MessageFormatter.format_timestamp(timestamp)})"
        if role == cls.ROLE_COMMAND:
            preloaded_for = cls.preload_window(entry)
            only = f", {preloaded_for} window only" if preloaded_for else ""
            return f"Command result ({MessageFormatter.format_timestamp(timestamp)}{only})"
        if role == cls.ROLE_NOTE:
            return f"Note ({MessageFormatter.format_timestamp(timestamp)})"
        return MessageFormatter.format_header(role.capitalize(), timestamp, entry.get("source"), entry.get("location"))

    @staticmethod
    def append_user_turn(messages: list[dict], content: str) -> None:
        """Append a user message, merging into the trailing turn if it is already a
        user turn with string content. Avoids consecutive same-role turns, which
        Gemini and other providers reject (history can end in a user turn — an
        unanswered incoming message or a system event)."""
        if messages and messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], str):
            messages[-1] = {"role": "user", "content": messages[-1]["content"] + "\n\n" + content}
        else:
            messages.append({"role": "user", "content": content})

    @staticmethod
    def remove_from_user_turns(messages: list[dict], *texts: str) -> int:
        """Remove each of *texts*, in order, from every user turn with string content; returns
        how many turns changed. The dicts are replaced, not mutated (the list may share them
        with another holder), and a turn left empty is dropped."""
        changed = 0
        for i in range(len(messages) - 1, -1, -1):
            content = messages[i].get("content")
            if messages[i].get("role") != "user" or not isinstance(content, str):
                continue
            stripped = content
            for text in texts:
                stripped = stripped.replace(text, "")
            if stripped == content:
                continue
            if stripped:
                messages[i] = {**messages[i], "content": stripped}
            else:
                del messages[i]
            changed += 1
        return changed

    @staticmethod
    def strip_thinking_instruction(messages: list[dict]) -> int:
        """Undo get_messages(extended_thinking=True) on an already built message list, with the
        separator append_user_turn put around the instruction: what is left is byte for byte
        what extended_thinking=False would have produced."""
        return MessageHistory.remove_from_user_turns(
            messages, "\n\n" + THINKING_TAG_INSTRUCTION, THINKING_TAG_INSTRUCTION + "\n\n", THINKING_TAG_INSTRUCTION)

    def add_context_message(self, role: str, content: str, timestamp: float,
                            source: str, location: str | None = None):
        """Add an external context message (e.g. iMessage) to history."""
        self._save_history_entry(role, content, timestamp, source, location)
        self._load_history()
        self._notify_changed()

    def add_context_messages(self, source: str, entries) -> int:
        """Store a feed's batch of FeedEntry rows under `source` with one reload and one
        subscriber notify for the whole batch; returns how many rows were new."""
        if not entries:
            return 0
        try:
            stored = FloxyDatabase.get_instance().add_history_entries(
                [(e.role, e.content, e.timestamp, e.location) for e in entries], source)
        except Exception as e:
            info(f"MessageHistory: batch of {len(entries)} entries from {source} not saved "
                 f"({type(e).__name__}: {e}) — the feed keeps its cursor and retries")
            raise
        if not stored:
            return 0    # all duplicates (the Indigo feed re-submits the day every poll): nothing changed
        self._load_history()
        self._notify_changed()
        return stored

    def loaded_toolkits(self, window: str | None = WINDOW_TEXT) -> dict[str, float]:
        """Toolkit name → timestamp of the `command` rows inside `window` (at or after the
        trim cutoff, not preloaded for another window) that carry a toolkit reference, so
        load_toolkit can point back at one the model still sees instead of sending it again.
        `window=None` counts the rows of every window: the delta tracking of a carried
        section follows the reference wherever it is, a voice preload included."""
        loaded = {}
        for entry in self.get_merged_entries():
            if entry["timestamp"] < self._trim_cutoff:
                continue
            preloaded_for = self.preload_window(entry)
            if window is not None and preloaded_for is not None and preloaded_for != window:
                continue
            for name in self._toolkit_names(entry):
                loaded[name] = entry["timestamp"]
        trace(f"MessageHistory: toolkits in the {window or 'any'} window: {sorted(loaded) or 'none'}")
        return loaded

    def add_command_result(self, result: str, cap: bool = True, source: str | None = None):
        """Persist a command round's result (the `result: …` text the LLM was shown) as a
        `command` row, ordered after the newest row so it follows the assistant round that
        ran the command even when the reply was instant. `cap=False` stores it whole (a
        toolkit reference; the cap is for web pages and log dumps). `source` is the
        preload stamp (preload_source) of a row one window alone renders; a command row
        never carries anything else there."""
        if not result or not result.strip():
            trace("MessageHistory: empty command result, nothing to persist")
            return
        if cap and len(result) > self.COMMAND_RESULT_CHARS:
            info(f"MessageHistory: command result truncated for history "
                 f"({len(result)} → {self.COMMAND_RESULT_CHARS} chars)")
            result = result[:self.COMMAND_RESULT_CHARS] + " […result truncated in history…]"
        entries = self.get_merged_entries()
        newest = entries[-1]["timestamp"] if entries else 0.0
        timestamp = max(time.time(), newest + 0.001)
        self._save_history_entry(self.ROLE_COMMAND, result, timestamp, source)
        trace(f"MessageHistory: command result persisted ({len(result)} chars{f', {source}' if source else ''})")
        self._load_history()
        self._notify_changed()

    def add_exchange(self, user: str | None, user_query: str | None, assistant_reply: str | None,
                     source: str | None = None, location: str | None = None,
                     timestamp: float | None = None, note: str | None = None):
        """Add a user-assistant exchange to the history.

        `timestamp` dates the user row to when the request sent it (its header carried that
        time, so the row must render the same) and `note` is the clock note sent in front of
        the query, stored as its own verbatim row just before it — both from a byte-prefix
        request (RequestProcessor._prepare_llm_context). Without them the row is dated now."""
        current_time = time.time()

        if user_query is not None:
            query_time = timestamp if timestamp is not None else current_time
            if query_time < self._trim_cutoff:
                # The window head moved past the request's start while it ran (a trim during a
                # long prefill); dated at its start the row would fall outside the window.
                trace(f"MessageHistory: request-time {query_time:.3f} predates the trim cutoff — dating the user row now")
                query_time = current_time
            if note:
                self._save_history_entry(self.ROLE_NOTE, note, query_time - 0.001)
            self._save_history_entry(user, user_query, query_time, source, location)

        if assistant_reply is not None:
            self._save_history_entry("assistant", assistant_reply, current_time + 0.001)

        self._load_history()
        self._notify_changed()
