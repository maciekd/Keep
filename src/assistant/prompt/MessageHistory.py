import json
import time
from datetime import datetime
from typing import Callable

from assistant.ChannelProfile import WINDOW_TEXT
from common.logging import fine, info, trace
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

    # The cut on a user or assistant row (pasted text, a forwarded mail, a thinking block). A
    # `system` row is the server's own words, sized by the code that writes it, and rendered
    # whole: the first-run brief is the opening turn of the FTUE conversation and lists the
    # setup steps in order — cut at this size it ended mid-step 4 on every later turn, and the
    # assistant enabled Photos next to iMessage, before iCloud was signed in.
    MAX_RENDER_CHARS = 2000
    # A command's result is stored as its own `command` row after the assistant round that
    # ran it, so the next turn still sees what list_integrations / config_help / a search
    # returned instead of only the assistant's reaction to it. Stored capped at this size
    # (a web page read or a log dump is far bigger) and rendered whole — MAX_RENDER_CHARS
    # would cut a list_integrations table in half. Once a later row supersedes it (a newer
    # browse snapshot) HistoryCompressor collapses it at render time, before the trim.
    ROLE_COMMAND = DB_ROLE_COMMAND
    COMMAND_RESULT_CHARS = 6000   # per round, shared by its commands
    # The newest turn's rounds render as the request sent them, up to this many chars together,
    # until a later turn's round lands (_fresh): the turn after an answer must still see what
    # the answer stood on. Cut to 1,000 chars a result at once, six results left the next turn
    # without the evidence and it retracted a correct answer as made up (prod 2026-11-15).
    # Memory only — the stored row is the capped one, and a restart renders that.
    FRESH_RESULT_CHARS = 30000
    CUT_MARKER = " […cut here in history only: the turn that ran this read all {chars:,} chars…]"
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
    # Before a trim drops any row, the bulky `command` rows of the window (this size and up: a
    # toolkit reference, a procedure, a page read — one command brings any of them back) are
    # collapsed to a one-line stub, oldest first, until the window is back at LOW. A 19.5KB
    # procedure row pushed the window past the ceiling and the trim evicted the exchange the
    # user had just had — the decision 30 s old, the reference that caused it kept (#135).
    # The collapse is sticky like the cutoff (_collapsed): a row collapsed once stays
    # collapsed while it is in the window, so the bytes hold still between trims. The fresh
    # rule applies to it as to the drop: under the ceiling a row younger than
    # TRIM_FRESH_SECONDS is the task's and stays whole. And like the drop it is all or
    # nothing: when the rows it may collapse leave the window over HIGH and the drop must
    # wait for its rows to age, nothing changes — collapsing the one row that had just aged
    # on every turn rewrote the window every turn (prod 2026-11-02, 44–48KB, one row a turn).
    TRIM_COLLAPSE_BYTES = 2 * 1024
    # A trim that evicts toolkit reference rows leaves one transient `system` stub naming the
    # evicted toolkits, at the newest evicted row's timestamp and exempt from the cutoff, so
    # the LLM knows the commands it saw there are gone — without it the model keeps using
    # the toolkit from memory and guesses at names (a live run emitted browse_go for
    # browse_open). The stub carries the rule itself in one line and otherwise only the
    # names, since a voice day evicts a dozen at once (12 one-line stubs, 2.2KB, 2026-10-12).
    # A name leaves the stub when its toolkit is loaded again; the stub goes at the next
    # trim, with the day's wipe, or when its last name leaves. A preloaded row never puts
    # its name in: the preloader sends the toolkit again on the next request of its window,
    # and a stub telling the model to load it again stood over the reload itself (ds4 looped
    # on load_toolkit("setup") four times over that pair of lines, 2026-10-22).
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
        self.trim_collapse_bytes = self.TRIM_COLLAPSE_BYTES
        # Timestamp of the oldest entry in the LLM window; entries older than this
        # are hidden from get_history_messages even if they'd fit. Only advances
        # when the window exceeds trim_high_bytes (see _hysteresis_trim).
        self._trim_cutoff = 0.0
        # Timestamps of the `command` rows a trim collapsed to a stub (TRIM_COLLAPSE_BYTES).
        self._collapsed: set[float] = set()
        # The uncut text of the newest turn's command rows, by row timestamp (FRESH_RESULT_CHARS),
        # and whether that turn has been answered — the next round then lets them go.
        self._fresh: dict[float, str] = {}
        self._fresh_answered = False
        # The TUI renders the window many times a second and nothing changes between renders,
        # so the trim's "nothing to do" traces log once per state: the window size whose trim
        # was last deferred, the size the collapse pass last looked at, the preloaded names
        # a trim last evicted.
        self._deferred_trim_at = 0
        self._preloaded_evicted: set[str] = set()
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
        self._collapsed = set()
        self._fresh = {}
        self._preloaded_evicted = set()
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
                            section: str | None = None, base: str | None = None,
                            first: bool = False) -> dict:
        """Add an in-memory-only history entry. NOT persisted to floxy.db; returns the entry.

        Used by PromptManager to inject section-diff notes (see
        PromptManager._emit_transient_section_update). `section` tags the entry with the
        section it describes and `base` is the section state the note's diff starts from —
        never rendered, it lets a trim's restate bridge the baseline to the oldest surviving
        note (oldest_delta_note). `first` puts the entry ahead of the other transient rows
        instead of after them, for a row that shares the window head's timestamp and must
        render before the rows there. Once in the window a note stays until the trim drops it:
        removing or rewording a row changes the window's bytes from that row on, and the
        prefix cache reads the window only while every byte before the breakpoint repeats.
        """
        entry = {
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "source": source,
            "location": location,
            "section": section,
            "base": base,
        }
        if first:
            self.transient_entries.insert(0, entry)
        else:
            self.transient_entries.append(entry)
        trace(f"MessageHistory: transient entry added ({role}, {len(content)} chars, source={source}"
              f"{', at the window head' if first else ''})")
        self._notify_changed()
        return entry

    @property
    def trim_cutoff(self) -> float:
        """The timestamp of the window's oldest row; 0 while nothing has been trimmed."""
        return self._trim_cutoff

    def oldest_delta_note(self, section: str) -> dict | None:
        """The oldest prompt-delta note of `section` still inside the window, or None. Its `base`
        is where the section's surviving chain of notes starts."""
        notes = [e for e in self.transient_entries
                 if e.get("source") == self.SOURCE_PROMPT_DELTA and e.get("section") == section
                 and e["timestamp"] >= self._trim_cutoff]
        return min(notes, key=lambda e: e["timestamp"], default=None)

    def withdraw_unanswered_note(self, entry: dict) -> bool:
        """Remove `entry` if it is still a transient row with no assistant row after it, and
        return whether it was removed. Such a row sits in the trailing user turn that no cached
        request has ended on: a chat request is answered by an assistant row and the autonomous
        check puts its breakpoint on the last assistant row (ClaudeLLM._message_cache_index), so
        rewriting it changes no byte the prefix cache holds."""
        if not any(e is entry for e in self.transient_entries):
            trace("MessageHistory: the note to withdraw is gone (trimmed or wiped)")
            return False
        if any(row["role"] == "assistant" and row["timestamp"] > entry["timestamp"] for row in self.history):
            trace("MessageHistory: an assistant row stands after the note — it stays")
            return False
        self.transient_entries = [e for e in self.transient_entries if e is not entry]
        self._notify_changed()
        return True

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
        self._collapsed = {ts for ts in self._collapsed if ts >= self._trim_cutoff}

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
        kept, held = [], set()
        for row in rows:
            entry = row[0]
            preloaded_for = cls.preload_window(entry)
            if preloaded_for is not None and preloaded_for != window:
                continue
            if entry.get("source") == cls.SOURCE_PROMPT_DELTA:
                toolkit = CARRIED_SECTIONS.get(entry.get("section"))
                if toolkit is not None and toolkit not in held:
                    continue
            held.update(cls._toolkit_names(entry, row[2]))   # a collapsed reference holds nothing
            kept.append(row)
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
        rendered = self._with_fresh_results(self._render_entries(self.get_merged_entries(), self.max_render_chars))
        return self.visible(self._hysteresis_trim(self._compressor.compress(rendered)), window)

    def _with_fresh_results(self, rendered: list[tuple[dict, str, str]]) -> list[tuple[dict, str, str]]:
        """`rendered` with the newest turn's command rows uncut (_fresh). A row goes back to
        its stored, capped text for good — so it changes once — when it left the window or
        when uncut it would put the window where the trim acts: over the ceiling, or over the
        high watermark once it is older than trim_fresh_seconds. Oldest first."""
        if not self._fresh:
            return rendered
        in_window = {entry["timestamp"] for entry, _, _ in rendered if entry["timestamp"] >= self._trim_cutoff}
        for timestamp in [t for t in self._fresh if t not in in_window or t in self._collapsed]:
            del self._fresh[timestamp]
            trace("MessageHistory: an uncut command result left the window or was collapsed — let go")
        stored = {entry["timestamp"]: len(content.encode("utf-8")) for entry, _, content in rendered}
        total = sum(len(self._as_collapsed(row)[2].encode("utf-8")) for row in rendered
                    if row[0]["timestamp"] >= self._trim_cutoff and not self._is_eviction_stub(row[0]))
        total += sum(len(text.encode("utf-8")) - stored[timestamp] for timestamp, text in self._fresh.items())
        now = time.time()
        for timestamp in sorted(self._fresh):
            young = now - timestamp < self.trim_fresh_seconds
            limit = self.trim_ceiling_bytes if young else self.trim_high_bytes
            if total <= limit:
                break
            extra = len(self._fresh.pop(timestamp).encode("utf-8")) - stored[timestamp]
            total -= extra
            info(f"MessageHistory: the uncut command result of {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')} "
                 f"would hold the window over {limit // 1024}KB — rendering its capped row from now on "
                 f"({extra / 1024:.1f}KB less)")
        return [(entry, role, self._fresh.get(entry["timestamp"], content)) if entry["role"] == self.ROLE_COMMAND
                else (entry, role, content) for entry, role, content in rendered]

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

        Over the high watermark the bulky command rows are collapsed first (see
        TRIM_COLLAPSE_BYTES); rows are dropped only when that leaves the window over high.
        The window changes all at once or not at all: a collapse that leaves it over high
        is committed only together with the drop, and when the drop waits for its rows to
        age, so does the collapse — every change rewrites the cached window behind it.
        """
        stub = next((entry for entry, _, _ in rendered if self._is_eviction_stub(entry)), None)
        rendered = [item for item in rendered if not self._is_eviction_stub(item[0])]
        kept = [self._as_collapsed(item) for item in rendered if item[0]["timestamp"] >= self._trim_cutoff]
        total = sum(len(content.encode("utf-8")) for _, _, content in kept)
        if total <= self.trim_high_bytes:
            return self._with_eviction_stub(stub, kept, dropped=[])

        collapsed, collapsed_total, planned = self._collapse_plan(kept, total)
        if collapsed_total <= self.trim_high_bytes:
            self._commit_collapse(planned, collapsed_total)
            return self._with_eviction_stub(stub, collapsed, dropped=[])

        # Over the high watermark even collapsed: walk newest→oldest to find the low-watermark cut.
        keep_from = len(collapsed)
        kept_bytes = 0
        for i in range(len(collapsed) - 1, -1, -1):
            size = len(collapsed[i][2].encode("utf-8"))
            if kept_bytes + size > self.trim_low_bytes:
                break
            kept_bytes += size
            keep_from = i
        keep_from = self._snap_to_boundary(collapsed, keep_from, self.trim_high_bytes)
        if keep_from == 0:
            trace("MessageHistory: window over high watermark but nothing trimmable — leaving it")
            self._commit_collapse(planned, collapsed_total)
            return self._with_eviction_stub(stub, collapsed, dropped=[])
        newest_dropped_age = time.time() - collapsed[keep_from - 1][0]["timestamp"]
        fresh = newest_dropped_age < self.trim_fresh_seconds
        if fresh and total <= self.trim_ceiling_bytes:
            if total != self._deferred_trim_at:   # the TUI renders the window many times a second
                self._deferred_trim_at = total
                trace(f"MessageHistory: window at {total / 1024:.1f}KB (high {self.trim_high_bytes // 1024}KB) but "
                      f"collapsing {len(planned)} bulky row(s) would leave it at {collapsed_total / 1024:.1f}KB and "
                      f"the newest row a trim would drop is {newest_dropped_age / 60:.1f}m old — waiting for it to "
                      f"age past {self.trim_fresh_seconds // 60}m or the window to hit the "
                      f"{self.trim_ceiling_bytes // 1024}KB ceiling")
            return self._with_eviction_stub(stub, kept, dropped=[])
        self._commit_collapse(planned, collapsed_total)
        total = collapsed_total
        kept_bytes = sum(len(content.encode("utf-8")) for _, _, content in collapsed[keep_from:])

        dropped = collapsed[:keep_from]
        kept = collapsed[keep_from:]
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

    # --- collapsed command rows (see TRIM_COLLAPSE_BYTES) ---------------------------------

    def _as_collapsed(self, row: tuple[dict, str, str]) -> tuple[dict, str, str]:
        """`row` rendered as its stub when a trim collapsed it, else as it is."""
        entry = row[0]
        if entry["timestamp"] not in self._collapsed:
            return row
        return (entry, row[1], self._collapsed_text(entry))

    def _collapse_plan(self, kept: list[tuple[dict, str, str]], total: int) -> tuple[list, int, list[tuple[dict, int]]]:
        """The window with its bulky command rows collapsed, oldest first, until it is back at
        the low watermark, its size, and the rows collapsed with their sizes. Under the ceiling
        a row younger than trim_fresh_seconds stays whole. A plan only: _commit_collapse
        remembers it once the trim decides the window changes."""
        now = time.time()
        over_ceiling = total > self.trim_ceiling_bytes
        kept = list(kept)
        planned = []
        for i, (entry, role, content) in enumerate(kept):
            if total <= self.trim_low_bytes:
                break
            if entry["role"] != self.ROLE_COMMAND or entry["timestamp"] in self._collapsed:
                continue
            size = len(content.encode("utf-8"))
            if size < self.trim_collapse_bytes:
                continue
            if now - entry["timestamp"] < self.trim_fresh_seconds and not over_ceiling:
                continue
            stub_text = self._collapsed_text(entry)
            kept[i] = (entry, role, stub_text)
            total -= size - len(stub_text.encode("utf-8"))
            planned.append((entry, size))
        return kept, total, planned

    def _commit_collapse(self, planned: list[tuple[dict, int]], total: int) -> None:
        """Remember the rows a plan collapsed (_collapsed), so they stay collapsed while in the window."""
        if not planned:
            return
        self._collapsed.update(entry["timestamp"] for entry, _ in planned)
        rows = ", ".join(f"{datetime.fromtimestamp(entry['timestamp']).strftime('%H:%M:%S')} ({size / 1024:.1f}KB)"
                         for entry, size in planned)
        info(f"MessageHistory: window over {self.trim_high_bytes // 1024}KB — collapsed {len(planned)} bulky "
             f"command row(s) to a stub before dropping anything: {rows}; window now {total / 1024:.1f}KB")

    @staticmethod
    def _collapsed_text(entry: dict) -> str:
        """The one line a collapsed command row renders as: a function of the stored row alone,
        so it never changes while the row is in the window. It must not start a line with a
        toolkit header: a collapsed reference is not loaded (`_toolkit_names`)."""
        content = entry["content"]
        lead = next((line.strip() for line in content.split("\n")
                     if line.strip() and line.strip() != "result:"), "")
        if len(lead) > 80:
            lead = lead[:80] + "…"
        return (f"[A {len(content.encode('utf-8')) / 1024:.1f}KB command result was trimmed from the window to "
                f"make room — it began \"{lead}\". Run the command again if it is needed.]")

    # --- toolkit-eviction stubs (see SOURCE_TOOLKIT_EVICTED) ------------------------------

    @classmethod
    def _is_eviction_stub(cls, entry: dict) -> bool:
        return entry.get("source") == cls.SOURCE_TOOLKIT_EVICTED

    @staticmethod
    def _toolkit_names(entry: dict, content: str | None = None) -> list[str]:
        """The toolkit names a `command` row's reference text carries (none for other rows).
        `content` is the row as rendered when that is what counts: a collapsed reference
        renders without its header and so holds no toolkit."""
        from assistant.commands.Toolkits import loaded_name
        if entry["role"] != MessageHistory.ROLE_COMMAND:
            return []
        text = entry["content"] if content is None else content
        return [name for name in (loaded_name(line) for line in text.split("\n")) if name]

    @staticmethod
    def _stub_text(names: list[str]) -> str:
        return (f"Toolkit references trimmed from the window: {', '.join(names)}. A toolkit loaded "
                f"again below is current; load the others again before using them.")

    def _with_eviction_stub(self, stub: dict | None, kept: list[tuple[dict, str, str]],
                            dropped: list[tuple[dict, str, str]]) -> list[tuple[dict, str, str]]:
        """The window with its one toolkit-eviction stub in front. The stub holds still between
        trims: it is the window's first row, and a byte of it changing (a name leaving when its
        toolkit was loaded again, as it used to; a name or the timestamp of a reference a
        collapse had just emptied, as it did for a day) is a cache miss on the whole window —
        so the text points at the reload rows below instead, and a collapsed reference is not
        the stub's business: its own one-line stub says the text is gone. A trim (`dropped`
        non-empty) retires the earlier trim's stub whole — a toolkit not reloaded within one
        window's worth of conversation is not in use — and leaves a fresh one naming every
        toolkit row it evicted."""
        in_window = {name for entry, _, content in kept for name in self._toolkit_names(entry, content)}
        names = [] if dropped or stub is None else list(stub["toolkits"])

        evicted: dict[str, float] = {}
        preloaded = set()
        for entry, _, _ in dropped:
            for name in self._toolkit_names(entry):
                if name in in_window:
                    continue
                if self.preload_window(entry) is not None:
                    preloaded.add(name)
                    continue
                evicted[name] = max(evicted.get(name, 0.0), entry["timestamp"])
        preloaded -= set(evicted)
        if preloaded != self._preloaded_evicted:
            self._preloaded_evicted = preloaded
            if preloaded:
                trace(f"MessageHistory: trim evicted the preloaded reference(s) of "
                      f"{', '.join(sorted(preloaded))} — the preloader brings them back, no stub")
        if evicted:
            names = sorted(set(names) | set(evicted))
        timestamp = max(evicted.values()) if evicted else (stub["timestamp"] if stub else 0.0)

        if stub is not None and (names != stub["toolkits"] or timestamp != stub["timestamp"]):
            self.transient_entries = [e for e in self.transient_entries if e is not stub]
            stub = None
        if names and stub is None:
            # Logged here, not where `evicted` is found: the window is rendered on every
            # request and many times a second by the TUI, and the evicted rows stay evicted.
            info(f"MessageHistory: trim evicted the toolkit reference(s) of {', '.join(names)} "
                 f"— one stub names them at the newest row's place")
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

        if role == "system":
            # System/home-automation events ride on the user side: the message API has no
            # in-timeline system role (only the top-level system prompt), so external input
            # gets the "user" role and the "System (...)" text prefix carries the semantics.
            # Never cut (MAX_RENDER_CHARS).
            return ("user", f"System ({MessageFormatter.format_timestamp(timestamp)}): {content}")

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
    def load_fixture_messages(path: str, rewrite: Callable[[str], str] | None = None) -> list[dict]:
        """Render a JSON history fixture into LLM message turns (prompt test suite).

        The fixture is a list of entries shaped like FloxyDatabase.get_recent_history()
        rows ({role, content, timestamp, source, location}), with timestamp as a local
        'YYYY-MM-DD HH:MM:SS' string converted to Unix time here. The result is rendered
        once and held fixed, so it sends the exact same turns production sends — without
        the live, mutable history accumulating per-test traffic. `rewrite` is applied to
        every assistant turn's content before rendering: the PromptManager renders the
        calls in them in the active command format.
        """
        with open(path, "r") as f:
            raw = json.load(f)

        entries = []
        for entry in raw:
            ts = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S").timestamp()
            content = entry["content"]
            if rewrite is not None and entry["role"] == "assistant":
                content = rewrite(content)
            entries.append({
                "role": entry["role"],
                "content": content,
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
            if entry["timestamp"] < self._trim_cutoff or entry["timestamp"] in self._collapsed:
                continue
            preloaded_for = self.preload_window(entry)
            if window is not None and preloaded_for is not None and preloaded_for != window:
                continue
            for name in self._toolkit_names(entry):
                loaded[name] = entry["timestamp"]
        fine(f"MessageHistory: toolkits in the {window or 'any'} window: {sorted(loaded) or 'none'}")
        return loaded

    def _history_shares(self, parts: list[str], budget: int) -> list[int]:
        """How many chars of each result a row of `budget` chars keeps: the short ones whole,
        the long ones an equal share of what the short ones leave."""
        shares, left = [0] * len(parts), budget
        by_length = sorted(range(len(parts)), key=lambda i: len(parts[i]))
        for position, i in enumerate(by_length):
            shares[i] = min(len(parts[i]), left // (len(parts) - position))
            left -= shares[i]
        return shares

    def _capped(self, result: str, parts: list[str] | None, budget: int) -> tuple[str, list[str], list[int]]:
        """`result` with its parts cut to their shares of `budget`, each cut one ending in the
        marker; with the parts found in it and what each keeps."""
        found = [p for p in parts or [] if p in result] or [result]
        kept = self._history_shares(found, budget)
        for part, share in zip(found, kept):
            if share < len(part):
                result = result.replace(part, part[:share] + self.CUT_MARKER.format(chars=len(part)), 1)
        return result, found, kept

    def add_command_result(self, result: str, cap: bool = True, source: str | None = None,
                           parts: list[str] | None = None):
        """Persist a command round's result (the `result: …` text the LLM was shown) as a
        `command` row, ordered after the newest row so it follows the assistant round that
        ran the command even when the reply was instant. `cap=False` stores it whole (a
        toolkit reference; the cap is for web pages and log dumps). `parts` are the results
        of the round's commands as they stand in `result`: they share the cap
        (_history_shares), so one long page does not push the results after it out of the
        row, and each cut one ends in a marker saying the cut is the history's — read as a
        fetch that failed, it made a later turn retract figures the turn that ran the
        commands had read (prod 2026-11-15). Without parts the cap is on the whole text.
        A cut row still renders uncut while its turn is the newest (FRESH_RESULT_CHARS).
        `source` is the preload stamp (preload_source) of a row one
        window alone renders; a command row never carries anything else there."""
        if not result or not result.strip():
            trace("MessageHistory: empty command result, nothing to persist")
            return
        if source is None and self._fresh_answered:
            # A preload is no round: it lands in front of the very turn the uncut rows are kept for.
            if self._fresh:
                info(f"MessageHistory: a later turn's round — the {len(self._fresh)} uncut command result(s) "
                     f"of the turn before render capped from now on")
            self._fresh, self._fresh_answered = {}, False
        fresh = None
        if cap:
            whole = len(result)
            fresh = self._capped(result, parts, self.FRESH_RESULT_CHARS)[0]
            result, found, kept = self._capped(result, parts, self.COMMAND_RESULT_CHARS)
            if len(result) != whole:
                info(f"MessageHistory: command result cut for history ({whole} → {len(result)} chars; "
                     f"{len(found)} result(s) keep {kept} of {[len(p) for p in found]})")
        entries = self.get_merged_entries()
        newest = entries[-1]["timestamp"] if entries else 0.0
        timestamp = max(time.time(), newest + 0.001)
        self._save_history_entry(self.ROLE_COMMAND, result, timestamp, source)
        trace(f"MessageHistory: command result persisted ({len(result)} chars{f', {source}' if source else ''})")
        self._load_history()
        if fresh is not None and fresh != result:
            self._keep_fresh(timestamp, fresh)
        self._notify_changed()

    def _keep_fresh(self, timestamp: float, text: str):
        """Render the row stored at `timestamp` as `text`, its uncut form, while its turn is
        the newest. The turn's rounds share FRESH_RESULT_CHARS: the oldest go back to their
        capped rows when a new one takes them over it."""
        self._fresh[timestamp] = text
        while len(self._fresh) > 1 and sum(len(t) for t in self._fresh.values()) > self.FRESH_RESULT_CHARS:
            del self._fresh[min(self._fresh)]
            trace("MessageHistory: the turn's uncut results are over their share — the oldest renders capped")
        info(f"MessageHistory: the row is cut, the window renders it uncut ({len(text)} chars) until a later "
             f"turn's round lands; {len(self._fresh)} uncut row(s), {sum(len(t) for t in self._fresh.values())} chars")

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
        if self._fresh and not self._fresh_answered:
            trace(f"MessageHistory: the turn is answered — its {len(self._fresh)} uncut command result(s) "
                  f"stay until a later turn's round")
        self._fresh_answered = True

        self._load_history()
        self._notify_changed()
