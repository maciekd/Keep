import json
import time
from datetime import datetime
from typing import Callable

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
    # A trim that evicts a toolkit reference row leaves a one-line transient `system` stub
    # at the row's timestamp, exempt from the cutoff, so the LLM knows the commands it saw
    # there are gone and how to get them back — without it the model keeps using the toolkit
    # from memory and guesses at names (a live run emitted browse_go for browse_open). The
    # stub goes when the toolkit is loaded again, at the next trim, or with the day's wipe.
    SOURCE_TOOLKIT_EVICTED = "toolkit_evicted"
    # Before the cut moves the window head, the toolkit reference rows in the window go
    # first (superseded copies, then oldest first, the fresh-row rule above applying): a
    # reference is reloadable with load_toolkit, the conversation rows the cut would drop
    # instead are not, and the head stays while the references alone bring the window under
    # the high watermark — a 7 KB home reference is a fifth of it. An
    # evicted row stays in floxy.db and the TUI; the window hides it (_evicted_rows) and
    # leaves its stub, and loaded_toolkits no longer counts it so the preloader sends it again.
    # The first row of a trimmed window is the trim note, "N older entries before HH:MM were
    # trimmed — use show_history for that period", frozen at the trim like the cutoff, so the
    # LLM reaches for show_history instead of trusting a note the trim took the context of.
    SOURCE_TRIM_NOTE = "trim_note"
    # The short row a background job leaves when it finishes — self_improvement (its PR),
    # self_analysis (its first line), a scheduled wakeup — is exempt from the trim for the
    # rest of the day (the window only holds today's rows): on 2026-10-11 the assistant
    # believed a self-improvement job was still running two hours after its completion row
    # had been trimmed away. A pinned row older than the window head stands in front of it.
    SOURCE_SELF_IMPROVEMENT = "self_improvement"
    SOURCE_SELF_ANALYSIS = "self_analysis"
    SOURCE_WAKEUP = "wakeup"
    PINNED_SOURCES = frozenset({SOURCE_SELF_IMPROVEMENT, SOURCE_SELF_ANALYSIS, SOURCE_WAKEUP})

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
        # Timestamps of the toolkit reference rows a trim evicted ahead of the cut (SOURCE_TRIM_NOTE above).
        self._evicted_rows: set[float] = set()
        # What the trim note says, frozen at the last trim: how many rows are out of the window
        # and the oldest one's time. Counted once so the note's bytes, the window's first, stay put.
        self._trimmed_count = 0
        self._trimmed_from = 0.0
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
        self._evicted_rows = set()
        self._trimmed_count = 0
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

        Used by PromptManager to inject section-diff notes between full prompt
        rebuilds (see PromptManager._emit_transient_section_update). `section` tags
        the entry with the section it describes so a later delta for that same
        section can supersede it (clear_transient_entries_for_section).
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

    def clear_transient_entries(self, keep_sections: tuple[str, ...] = ()):
        """Drop the in-memory transient entries. Called by PromptManager on full rebuild
        (the new prompt now embeds the section data, so the deltas are obsolete), except the
        notes of `keep_sections` — the sections a toolkit carries, which no prompt embeds —
        and the toolkit-eviction stubs, which describe the history window, not the prompt."""
        if not self.transient_entries:
            return
        kept = [e for e in self.transient_entries
                if e.get("section") in keep_sections or self._is_eviction_stub(e)]
        count = len(self.transient_entries) - len(kept)
        if not count:
            trace(f"MessageHistory: nothing to clear, all {len(kept)} transient entries are carried-section notes")
            return
        self.transient_entries = kept
        info(f"MessageHistory: cleared {count} transient entries" + (f", kept {len(kept)}" if kept else ""))
        self._notify_changed()

    def clear_transient_entries_for_section(self, section: str) -> int:
        """Drop in-memory transient notes tagged for a given section; return the count.

        PromptManager calls this before emitting a fresh delta for a section (and when
        a section reverts to its baseline) so only the latest delta per section is ever
        shown to the LLM. Each delta is a diff against the frozen baseline, so an older
        one describes a stale intermediate state — append-only entries would otherwise
        keep those visible until the next full rebuild.
        """
        if not self.transient_entries:
            return 0
        before = len(self.transient_entries)
        self.transient_entries = [e for e in self.transient_entries if e.get("section") != section]
        dropped = before - len(self.transient_entries)
        if dropped:
            info(f"MessageHistory: cleared {dropped} stale transient entries for section '{section}'")
            self._notify_changed()
        return dropped

    def get_history_messages(self, max_bytes: int | None = None) -> list[dict]:
        """Render history as standard LLM message items (role/content dicts).

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
        return self._finalize_messages([(role, content) for _, role, content in self._window()])

    def _window(self) -> list[tuple[dict, str, str]]:
        """The live history window as (entry, role, content) rows: every entry rendered,
        superseded rows compressed, the hysteresis trim applied. get_history_messages sends
        it and render_for_display shows it, so the Prompt tab never drifts from the request."""
        rendered = self._compressor.compress(self._render_entries(self.get_merged_entries(), self.max_render_chars))
        return self._hysteresis_trim(rendered)

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

        Over the high watermark the toolkit reference rows go first (_evict_toolkit_rows);
        the cut moves the head only when they alone do not bring the window back under it. Between high and
        trim_ceiling_bytes the cut waits while the newest row it would drop is younger than
        trim_fresh_seconds (see TRIM_CEILING_BYTES); the age is checked after snapping, since
        the snap can move the cut and with it that row. The pinned rows (PINNED_SOURCES) and
        the eviction stubs are outside the budget and the cut, and are put back by _assemble.
        """
        stubs = [item for item in rendered if self._is_eviction_stub(item[0])]
        pinned = [item for item in rendered if self._is_pinned(item[0])]
        rows = [item for item in rendered
                if not self._is_eviction_stub(item[0]) and not self._is_pinned(item[0])]
        rows = [item for item in rows if not self._is_evicted(item[0])]
        kept = [item for item in rows if item[0]["timestamp"] >= self._trim_cutoff]
        total = sum(len(content.encode("utf-8")) for _, _, content in kept)
        if total <= self.trim_high_bytes:
            return self._assemble(stubs, pinned, kept, dropped=[])

        evicted, kept, total = self._evict_toolkit_rows(kept, total)
        if evicted and total <= self.trim_high_bytes:
            return self._assemble(stubs, pinned, kept, dropped=evicted)

        # Still over the high watermark: walk newest→oldest to find the low-watermark cut.
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
            return self._assemble(stubs, pinned, kept, dropped=evicted)
        newest_dropped_age = time.time() - kept[keep_from - 1][0]["timestamp"]
        fresh = newest_dropped_age < self.trim_fresh_seconds
        if fresh and total <= self.trim_ceiling_bytes:
            if total != self._deferred_trim_at:   # the TUI renders the window many times a second
                self._deferred_trim_at = total
                trace(f"MessageHistory: window at {total / 1024:.1f}KB (high {self.trim_high_bytes // 1024}KB) but "
                      f"the newest row a trim would drop is {newest_dropped_age / 60:.1f}m old — waiting for it to "
                      f"age past {self.trim_fresh_seconds // 60}m or the window to hit the "
                      f"{self.trim_ceiling_bytes // 1024}KB ceiling")
            return self._assemble(stubs, pinned, kept, dropped=evicted)
        kept_bytes = sum(len(content.encode("utf-8")) for _, _, content in kept[keep_from:])

        dropped = kept[:keep_from]
        kept = kept[keep_from:]
        self._trim_cutoff = kept[0][0]["timestamp"]
        why = (f"ceiling {self.trim_ceiling_bytes // 1024}KB, newest dropped row only {newest_dropped_age / 60:.1f}m old"
               if fresh else f"high {self.trim_high_bytes // 1024}KB")
        info(f"MessageHistory: window hit {total / 1024:.1f}KB "
             f"({why}) — trimmed {len(dropped)} entries, "
             f"kept {len(kept)} ({kept_bytes / 1024:.1f}KB), new cutoff "
             f"{datetime.fromtimestamp(self._trim_cutoff).strftime('%H:%M:%S')}")
        dropped_notes = [entry for entry, _, _ in dropped if entry.get("source") == "prompt_delta"]
        if dropped_notes:
            sections = {entry.get("section") for entry in dropped_notes} - {None}
            self.dropped_delta_sections |= sections
            info(f"MessageHistory: trim dropped {len(dropped_notes)} prompt-delta note(s) of "
                 f"{sorted(sections)} — PromptManager restates those sections before the next send")
        self._freeze_trim_note(rows, self._trim_cutoff)
        return self._assemble(stubs, pinned, kept, evicted + dropped)

    def _evict_toolkit_rows(self, kept: list[tuple[dict, str, str]], total: int):
        """The first pass of a trim over the high watermark: the toolkit reference rows of the
        window until it is back under the high watermark — the superseded copies (a toolkit
        with a newer row in the window) always, then the newest copies oldest first. Between
        high and the ceiling a row younger than trim_fresh_seconds stays (the toolkit a task
        loaded a minute ago is the one it is using). When the references alone do not bring
        the window under the watermark the cut follows, and it keeps the tail — so the newest
        copies are left to it rather than evicted for nothing. Returns (evicted, kept, total)."""
        candidates = [item for item in kept if self._toolkit_names(item[0])]
        if total <= self.trim_ceiling_bytes:
            now = time.time()
            candidates = [item for item in candidates if now - item[0]["timestamp"] >= self.trim_fresh_seconds]
        if not candidates:
            trace("MessageHistory: window over high watermark and no toolkit reference row to evict ahead of the cut")
            return [], kept, total
        newest = {}
        for entry, _, _ in kept:
            for name in self._toolkit_names(entry):
                newest[name] = max(newest.get(name, 0.0), entry["timestamp"])
        size = lambda item: len(item[2].encode("utf-8"))
        superseded = [item for item in candidates
                      if all(newest[name] > item[0]["timestamp"] for name in self._toolkit_names(item[0]))]
        latest = sorted((item for item in candidates if item not in superseded), key=lambda item: item[0]["timestamp"])
        evicted = list(superseded)
        total -= sum(size(item) for item in superseded)
        for item in latest:
            if total <= self.trim_high_bytes:
                break
            evicted.append(item)
            total -= size(item)
        if total > self.trim_high_bytes and len(evicted) > len(superseded):
            total += sum(size(item) for item in evicted[len(superseded):])
            evicted = superseded
        if not evicted:
            trace("MessageHistory: the toolkit references alone would not bring the window under the high "
                  "watermark — left to the cut")
            return [], kept, total
        self._evicted_rows |= {item[0]["timestamp"] for item in evicted}
        kept = [item for item in kept if item not in evicted]
        names = sorted({name for entry, _, _ in evicted for name in self._toolkit_names(entry)})
        info(f"MessageHistory: window over high watermark — evicted {len(evicted)} toolkit reference row(s) "
             f"({', '.join(names)}) ahead of the cut, {total / 1024:.1f}KB left"
             + (", the cut follows" if total > self.trim_high_bytes else ", the head stays"))
        return evicted, kept, total

    def _is_evicted(self, entry: dict) -> bool:
        return entry["role"] == self.ROLE_COMMAND and entry["timestamp"] in self._evicted_rows

    @classmethod
    def _is_pinned(cls, entry: dict) -> bool:
        return entry.get("source") in cls.PINNED_SOURCES

    # --- the trim note (see SOURCE_TRIM_NOTE) ----------------------------------------------

    def _freeze_trim_note(self, rows: list[tuple[dict, str, str]], cutoff: float):
        """Count the rows before the cutoff once the cut has moved it, so the note keeps saying
        the same thing until the next trim (an evicted toolkit row has its stub instead)."""
        out = [entry for entry, _, _ in rows if entry["timestamp"] < cutoff]
        self._trimmed_count = len(out)
        self._trimmed_from = min((entry["timestamp"] for entry in out), default=cutoff)
        trace(f"MessageHistory: the trim note now covers {self._trimmed_count} row(s) since "
              f"{datetime.fromtimestamp(self._trimmed_from).strftime('%H:%M:%S')}")

    def _trim_note(self) -> tuple[dict, str, str] | None:
        """The window's first row once a trim has happened, dated just before the head."""
        if not self._trimmed_count:
            return None
        cutoff = datetime.fromtimestamp(self._trim_cutoff)
        since = datetime.fromtimestamp(self._trimmed_from)
        noun = "entry" if self._trimmed_count == 1 else "entries"
        text = (f"{self._trimmed_count} older {noun} before {cutoff.strftime('%H:%M')} were trimmed from this "
                f"conversation — use show_history(\"{since.strftime('%Y-%m-%d %H:%M')}\", "
                f"\"{cutoff.strftime('%Y-%m-%d %H:%M')}\") for that period.")
        entry = {"role": "system", "content": text, "timestamp": self._trim_cutoff - 0.001,
                 "source": self.SOURCE_TRIM_NOTE, "location": None, "section": None}
        return (entry, *self._render_entry(entry, self.max_render_chars))

    def _assemble(self, stubs: list[tuple[dict, str, str]], pinned: list[tuple[dict, str, str]],
                  kept: list[tuple[dict, str, str]], dropped: list[tuple[dict, str, str]]) -> list[tuple[dict, str, str]]:
        """The window in order: the trim note, then the stubs and pinned rows merged into the
        kept rows by time — those older than the head in front of it."""
        window = self._with_eviction_stubs(stubs, kept, dropped)
        if pinned:
            window = sorted(window + pinned, key=lambda item: item[0]["timestamp"])
        # An orphaned reply at the head (its query cut, see _snap_to_boundary) goes before the
        # note is put in front of it — _finalize_messages would no longer pop it.
        while window and window[0][1] != "user":
            window.pop(0)
        note = self._trim_note()
        return [note] + window if note else window

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

    @classmethod
    def _stub_text(cls, name: str) -> str:
        return (f"The {name} toolkit reference loaded here was trimmed from the history window; "
                f"its commands are no longer documented in it. load_toolkit(\"{name}\") sends it again.")

    def _with_eviction_stubs(self, stubs: list[tuple[dict, str, str]], kept: list[tuple[dict, str, str]],
                             dropped: list[tuple[dict, str, str]]) -> list[tuple[dict, str, str]]:
        """The window with its toolkit-eviction stubs at their rows' places. A stub whose toolkit is back in
        the window is retired; a trim (`dropped` non-empty) retires the earlier trim's stubs —
        a toolkit not reloaded within one window's worth of conversation is not in use — and
        leaves a fresh one for every toolkit row it evicted."""
        in_window = {name for entry, _, _ in kept for name in self._toolkit_names(entry)}
        retired = [entry for entry, _, _ in stubs if entry["toolkit"] in in_window]
        if dropped:
            retired += [entry for entry, _, _ in stubs if entry["toolkit"] not in in_window]
        if retired:
            names = sorted({e["toolkit"] for e in retired})
            trace(f"MessageHistory: retiring the eviction stub(s) of {', '.join(names)}"
                  + (" — loaded again" if not dropped else " — not loaded again before the next trim"))
            self.transient_entries = [e for e in self.transient_entries if e not in retired]
            stubs = [item for item in stubs if item[0] not in retired]

        evicted: dict[str, float] = {}
        for entry, _, _ in dropped:
            for name in self._toolkit_names(entry):
                if name not in in_window:
                    evicted[name] = max(evicted.get(name, 0.0), entry["timestamp"])
        if evicted:
            info(f"MessageHistory: trim evicted the toolkit reference(s) of {', '.join(sorted(evicted))} "
                 f"— leaving a stub for each at its row's place")
            new_stubs = [{"role": "system", "content": self._stub_text(name), "timestamp": timestamp,
                          "source": self.SOURCE_TOOLKIT_EVICTED, "location": None, "section": None,
                          "toolkit": name}
                         for name, timestamp in sorted(evicted.items(), key=lambda item: item[1])]
            self.transient_entries = self.transient_entries + new_stubs
            stubs = stubs + [(entry, *self._render_entry(entry, self.max_render_chars)) for entry in new_stubs]
        if not stubs:
            return kept
        return sorted(stubs + kept, key=lambda item: item[0]["timestamp"])

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

    def render_for_display(self) -> str:
        """Render the LLM history window as markdown for the console PromptTab viewer.

        The rows are the window get_history_messages() sends (_window: rendered,
        compressed, trimmed), so every body is byte for byte what the LLM receives —
        a superseded browse snapshot shows collapsed, a long turn shows cut. Each row
        gets an h3 header (name/time/source) so MarkdownTabView colours it; the
        entries out of the window are what the trim note at its head says they are.
        """
        if not self.get_merged_entries():
            return "_No conversation history yet._"
        return "\n\n".join(f"### {self._display_header(entry)}\n{content}" for entry, _, content in self._window())

    @classmethod
    def _display_header(cls, entry: dict) -> str:
        role = entry["role"]
        timestamp = entry["timestamp"]
        if role == "assistant":
            return MessageFormatter.format_assistant_header(timestamp)
        if role == "system":
            return f"System ({MessageFormatter.format_timestamp(timestamp)})"
        if role == cls.ROLE_COMMAND:
            return f"Command result ({MessageFormatter.format_timestamp(timestamp)})"
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

    def loaded_toolkits(self) -> dict[str, float]:
        """Toolkit name → timestamp of the `command` rows inside the LLM's window (at or after
        the trim cutoff) that carry a toolkit reference, so load_toolkit can point back at
        one the model still sees instead of sending it again."""
        loaded = {}
        for entry in self.get_merged_entries():
            if entry["timestamp"] < self._trim_cutoff or self._is_evicted(entry):
                continue
            for name in self._toolkit_names(entry):
                loaded[name] = entry["timestamp"]
        trace(f"MessageHistory: toolkits in the window: {sorted(loaded) or 'none'}")
        return loaded

    def add_command_result(self, result: str, cap: bool = True):
        """Persist a command round's result (the `result: …` text the LLM was shown) as a
        `command` row, ordered after the newest row so it follows the assistant round that
        ran the command even when the reply was instant. `cap=False` stores it whole (a
        toolkit reference; the cap is for web pages and log dumps)."""
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
        self._save_history_entry(self.ROLE_COMMAND, result, timestamp)
        trace(f"MessageHistory: command result persisted ({len(result)} chars)")
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
