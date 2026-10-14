import base64
import difflib
import re
import threading
import time
import os
import traceback
import uuid
from datetime import datetime

from . import PromptSectionGenerator as SectionGenerators
from .TimeOfDay import TimeOfDay
from .sections.MemorySectionGenerator import MemorySectionGenerator
from .sections.ProceduresSectionGenerator import ProceduresSectionGenerator
from .sections.TrialSectionGenerator import TrialSectionGenerator
from .sections.WatchListSectionGenerator import WatchListSectionGenerator
from .sections.IntegrationsSectionGenerator import IntegrationsSectionGenerator
from .sections.VaultSectionGenerator import VaultSectionGenerator
from .sections.HomeSectionGenerator import HomeSectionGenerator
from .sections.WebSectionGenerator import WebSectionGenerator
from .sections.CalendarSectionGenerator import CalendarSectionGenerator
from .sections.CarSectionGenerator import CarSectionGenerator
from assistant import Destinations
from assistant.ChannelProfile import WINDOW_TEXT
from .MessageHistory import MessageHistory, THINKING_TAG_INSTRUCTION
from common.FloxyDatabase import FloxyDatabase
from common.ConfigManager import config
from common.logging import error, info, trace, fine
from common.Paths import RES_DIR, SRC_DIR

# Frames from these AIServer/src packages count as "ours" in the section-update trigger trace.
_OWN_PACKAGE_DIRS = tuple(f"{SRC_DIR / pkg}/" for pkg in ("assistant", "common", "interfaces", "integrations"))


class PromptManager:
    """Assembles the live system prompt from its template and sections and builds the LLM message list.

    Section updates are folded in as small transient history notes where possible so the cached
    system-prompt bytes stay stable; a full rebuild happens only on cold start, large diffs or
    once per cache TTL.
    """

    # When a section updates: if the unified diff against the baseline (the bytes
    # baked into the currently-active prompt) is small enough, emit it as a
    # transient history note instead of rebuilding the prompt. Keeps the cached
    # system prompt bytes stable so Anthropic's 1h ephemeral cache stays warm.
    _DIFF_SIZE_THRESHOLD = 600
    # Hard ceiling on how long we'll keep delta-patching before forcing a full rebuild.
    # A rebuild changes the system prompt bytes, and with them the prefix of every cached
    # history turn behind it, so it must come rarer than the hourly autonomous check for
    # that check (and a chat resumed within the hour) to read its history instead of
    # re-writing it; the notes carry the section changes in between.
    _FULL_REBUILD_INTERVAL_S = 3 * 3600
    # A carried section (Toolkits.CARRIED_SECTIONS) is never re-baked into the prompt, so its
    # notes accumulate until the LLM loads the toolkit again; past this many chars of notes
    # one pointer note replaces them and tracking stops until the next load.
    _CARRIED_NOTES_BUDGET = 3000
    # {PHOTO_COUNT} refresh cadence. The live count creeps up as photos sync, and
    # re-reading it on every rebuild changes the prompt bytes for no user-visible
    # benefit. Fetch once at server start, then at most daily; rendered with a ~
    # prefix since the value is approximate between refreshes.
    _PHOTO_COUNT_REFRESH_S = 24 * 3600

    def __init__(self, test_suite: bool = False, section_generators=None):
        """`section_generators` are the integration folders' generator classes to run beside the
        core's; the discovered ones (PromptSectionGenerator.discover) when None."""
        self.prompt = ""
        self._messages = None
        self.test_suite = test_suite
        self._prompt_lock = threading.Lock()
        self._last_assembly_time = 0
        self._on_prompt_assembled_callbacks = []
        # The history window (ChannelProfile.window) the last request rendered: the one the
        # pre-heats warm, since the next request is most likely on the same channel.
        self._last_window = WINDOW_TEXT
        # Snapshot of each section's content at the time of the last full assembly —
        # the bytes baked into the currently-active prompt. Used to detect a cold
        # start (no baseline → rebuild).
        self._section_baselines: dict[str, str] = {}
        # The last content the LLM has actually been told about for each section:
        # the baseline if untouched, otherwise the `+` side of the most recent
        # delta note we emitted. Each new delta diffs against THIS (not the frozen
        # baseline), so notes form an append-only changelog — a later change adds a
        # fresh note instead of re-stating earlier ones under a new timestamp.
        self._section_current: dict[str, str] = {}
        # Per carried section: the chars of notes sent since its last load_toolkit (the
        # _CARRIED_NOTES_BUDGET meter) and whether a "load it again" pointer is out. Kept here
        # rather than read off the history's notes, which stay in the window once sent.
        self._carried_note_chars: dict[str, int] = {}
        self._carried_pointer_out: set[str] = set()
        # Cached {PHOTO_COUNT} value and when it was fetched (see _PHOTO_COUNT_REFRESH_S).
        self._photo_count: int | None = None
        self._photo_count_fetched_at = 0.0

        # Initialize section generators. Note: the current time/date is intentionally
        # NOT a section. It used to be (`{TIME_DATE}`), but it sat in the volatile
        # tail and changed every minute — invalidating the 5-min message-level
        # cache on every turn. It's now appended to the trailing user message in
        # get_messages() instead, where it lives in the uncached suffix and never
        # invalidates anything.
        # The generators that run only while their integration is on (key -> class, see
        # PromptSectionGenerator.enabled_by): built and started when the IntegrationMonitor
        # says so, stopped and dropped when it changes its mind (_on_integrations_changed).
        self._gated: dict[str, type] = {}
        self.sections = self._build_sections(
            SectionGenerators.discover() if section_generators is None else section_generators)
        self._assert_unique_section_names(self.sections)

        # Conversation history is no longer a prompt section — it's appended as
        # message items at request time (see get_messages).
        self.message_history = MessageHistory()

        # Prompt test suite: a fixed set of pre-rendered history turns loaded from a
        # fixture (see load_test_suite_history). Kept separate from message_history so
        # the live, mutable history never feeds the test prompt — test queries must not
        # accumulate into what subsequent tests see.
        self._test_history_messages: list[dict] = []
        self._test_toolkit_messages: list[dict] = []

        self._section_positions = self._compute_section_positions()

        if test_suite:
            # For test suite, just load test prompt immediately
            self._load_test_suite_prompt()
        else:
            # Start the core's section generators; each one's first fetch triggers a cheap
            # rebuild as it lands (a cold boot rebuilds a handful of times — fine). The gated
            # ones start when the IntegrationMonitor registers their integrations.
            info("Starting prompt section generators...")
            for name, section in self.sections.items():
                section.start()
            self._assemble_prompt()
            self._monitor().add_listener(self._on_integrations_changed)
            Destinations.add_listener(self._on_destinations_changed)

    def _on_destinations_changed(self):
        """An interface registered or dropped a destination: the {DESTINATIONS} line of the
        command block follows. Rare — at start and when the allow list changes — so the cached
        prompt is not churned by it."""
        self._assemble_prompt("destinations changed")

    # Where each section stands in `self.sections`: the order they are started and stopped in,
    # and the order of the {INTEGRATION_DATA} block. A section a new folder brings comes after
    # these, by key.
    SECTION_ORDER = ("memory", "memory_recent", "procedures", "watchlist", "integrations", "trial", "vault", "weather",
                     "precipitation", "home", "web", "calendar", "sonos", "music_zones", "car_info", "pool", "heat_pump", "phones")
    # The placeholder the data sections of the integrations fill, each under its own heading.
    DATA_BLOCK = "{INTEGRATION_DATA}"

    def _build_sections(self, generator_classes) -> dict:
        """The core's generators, plus one of each integration folder's under the key the
        class declares. A generator tied to an integration — a folder's, or a core one with
        `needs` — is not built here but noted in `_gated` for _on_integrations_changed. A key
        with two owners is an error naming both; a generator without a key, or whose
        constructor raises, is logged and left out."""
        memory = MemorySectionGenerator(self)
        sections = {
            'memory': memory,
            'memory_recent': memory.recent,
            'procedures': ProceduresSectionGenerator(self),
            'watchlist': WatchListSectionGenerator(self),
            'integrations': IntegrationsSectionGenerator(self),
            'trial': TrialSectionGenerator(self),
            'vault': VaultSectionGenerator(self),
            'web': WebSectionGenerator(self),
        }
        gated: dict[str, type] = {'home': HomeSectionGenerator, 'calendar': CalendarSectionGenerator,
                                  'car_info': CarSectionGenerator}
        for cls in generator_classes:
            if not cls.key:
                error(f"Section generator {cls.__module__}.{cls.__name__} declares no key, left out")
                continue
            if cls.key in sections or cls.key in gated:
                owner = gated[cls.key] if cls.key in gated else type(sections[cls.key])
                raise ValueError(f"Prompt section '{cls.key}' has two owners: "
                                 f"{owner.__module__}.{owner.__name__} and {cls.__module__}.{cls.__name__}")
            if cls.folder() is not None or cls.needs is not None:
                gated[cls.key] = cls
                continue
            try:
                sections[cls.key] = cls(self)
            except Exception as e:
                error(f"Section generator {cls.__module__}.{cls.__name__} could not be built, left out: {type(e).__name__}: {e}")
        self._gated = gated
        ordered = self._ordered(sections)
        trace(f"Prompt sections: {', '.join(ordered)}; waiting for their integration: {', '.join(gated)}")
        return ordered

    def section_classes(self) -> dict[str, type]:
        """Every section this manager knows, key -> generator class: the running ones and the
        gated ones waiting for their integration."""
        with self._prompt_lock:
            return {key: type(gen) for key, gen in self.sections.items()} | dict(self._gated)

    @classmethod
    def _ordered(cls, sections: dict) -> dict:
        order = {key: index for index, key in enumerate(cls.SECTION_ORDER)}
        return dict(sorted(sections.items(), key=lambda item: (order.get(item[0], len(order)), item[0])))

    def _monitor(self):
        """The IntegrationMonitor whose verdict gates the sections and the toolkit index; None
        in test-suite mode, where every toolkit is offered and no gated section runs."""
        if self.test_suite:
            return None
        from assistant.integration.IntegrationMonitor import IntegrationMonitor
        return IntegrationMonitor.get_instance()

    def _on_integrations_changed(self):
        """The IntegrationMonitor's listener: build and start the gated generators whose
        integration is on now, stop and drop the ones whose integration went off, then
        rebuild the prompt — the toolkit index follows the same verdict, so it is rebuilt even
        when no section changed. The generators are stopped outside the prompt lock: a stop
        joins the generator's thread, which may be waiting for that lock in on_section_updated."""
        monitor = self._monitor()
        started, stopped = [], []
        with self._prompt_lock:
            for key, cls in self._gated.items():
                running = key in self.sections
                wanted = cls.enabled_by(monitor)
                if wanted and not running:
                    try:
                        self.sections[key] = cls(self)
                    except Exception as e:
                        error(f"Section generator {cls.__module__}.{cls.__name__} could not be built, left out: "
                              f"{type(e).__name__}: {e}")
                        continue
                    started.append(key)
                elif running and not wanted:
                    stopped.append(self.sections.pop(key))
            if started or stopped:
                self.sections = self._ordered(self.sections)
            starting = [self.sections[key] for key in started]
        for section in stopped:
            section.stop()
        for section in starting:
            section.start()
        info(f"Integrations changed — sections started: {', '.join(started) or 'none'}, stopped: "
             f"{', '.join(s.section_name for s in stopped) or 'none'}; "
             f"waiting: {', '.join(k for k in self._gated if k not in self.sections) or 'none'}")
        self._assemble_prompt("integrations changed")

    def _compute_section_positions(self) -> dict[str, float]:
        """Map of section_key -> where the section stands in the prompt template: its own
        placeholder's position, or the {INTEGRATION_DATA} block's plus its rank in the block
        for a section with a heading. Sections placed nowhere are omitted (callers fall back
        to alpha order). Every section is mapped, the gated ones included, so the map is
        fixed for the life of the manager."""
        template_path = self._test_suite_template_path() if self.test_suite else "res/prompt.md"
        try:
            with open(template_path, "r") as f:
                template = f.read()
        except OSError as e:
            info(f"Could not read {template_path} for section order ({e})")
            return {}

        classes = {key: type(gen) for key, gen in self.sections.items()} | self._gated
        block_at = template.find(self.DATA_BLOCK)
        order = {key: index for index, key in enumerate(self.SECTION_ORDER)}
        positions = {}
        for section_key, cls in classes.items():
            idx = template.find("{" + section_key.upper() + "}")
            if idx >= 0:
                positions[section_key] = idx
            elif cls.heading and block_at >= 0:
                positions[section_key] = block_at + order.get(section_key, len(order)) / 1000
        if absent := [key for key in classes if key not in positions]:
            from assistant.commands import Toolkits
            info(f"Sections without a place in {template_path} (they reach the LLM through a toolkit, "
                 f"see Toolkits.CARRIED_SECTIONS): {', '.join(absent)}")
            # The test suite's own template is static: it places no section at all.
            if unreachable := [] if self.test_suite else [key for key in absent if key not in Toolkits.CARRIED_SECTIONS]:
                error(f"Sections that reach the LLM nowhere — no {{KEY}} placeholder or {self.DATA_BLOCK} block in "
                      f"{template_path} and no toolkit carries them (`sections:` in a toolkit.md): "
                      f"{', '.join(unreachable)}")
        return positions

    @staticmethod
    def _test_suite_template_path() -> str:
        """testsuite_prompt.md has a household's data baked in, so it is not part of the repository;
        a checkout without it runs the test suite on the production template with empty sections."""
        if os.path.exists("testsuite_prompt.md"):
            return "testsuite_prompt.md"
        info("No testsuite_prompt.md in this checkout, the test suite runs on res/prompt.md.")
        return "res/prompt.md"

    def _load_test_suite_prompt(self):
        """Load test suite prompt."""
        info("In testsuite, loading testsuite prompt.")
        template_path = self._test_suite_template_path()
        with open(template_path, "r") as file:
            template = file.read()
        # The data sections are baked into the test-suite prompt; only the config-driven
        # values and the toolkit index are substituted. The production template, the stand-in
        # when that file is absent, still has the section placeholders: they are blanked.
        sections = {} if template_path == "testsuite_prompt.md" else {key: "" for key in self.section_classes()}
        self.prompt = self._substitute(template, sections, photo_count=self._TEST_SUITE_PHOTO_COUNT)
        self._messages = [{"role": "system", "content": self.prompt}]
        self._test_toolkit_messages = self._all_toolkits_loaded()

    def _all_toolkits_loaded(self) -> list[dict]:
        """The prompt test suite checks a single reply, so it cannot afford a load round: every
        toolkit is put in front of the model as if one `load_toolkit(...)` call had fetched them
        all earlier in the conversation — the same two turns production produces."""
        from assistant.commands import Toolkits
        toolkits = Toolkits.TOOLKITS
        call = "load_toolkit(" + ", ".join(f'"{t.name}"' for t in toolkits) + ")"
        # A fixture like the rest of the test prompt: the count is baked, the DB may be absent.
        substitute = lambda text: self._substitute(text, {}, photo_count=self._TEST_SUITE_PHOTO_COUNT)
        result = "result: \n" + "\n\n".join(Toolkits.render(t, substitute) for t in toolkits)
        info(f"Test suite: all {len(toolkits)} toolkits appended to the history ({len(result)} chars)")
        return [{"role": "assistant", "content": call}, {"role": "user", "content": result}]

    @staticmethod
    def _wake_interval_minutes_str() -> str:
        """Single source of truth for the periodic wake-up cadence advertised to the
        LLM — mirrors the autonomous action manager's actual sleep interval."""
        from ..agents.AutonomousActionManager import PERIODIC_CHECK_INTERVAL_MINUTES
        return f"{PERIODIC_CHECK_INTERVAL_MINUTES:.0f}"

    @staticmethod
    def _get_trigger_trace() -> str:
        """Extract the two most relevant caller frames from the stack, skipping internal plumbing."""
        plumbing = {'_assemble_prompt', 'on_section_updated', 'notify_update',
                    '_update_content', '_start_monitoring_wrapper', '_get_trigger_trace'}
        frames = traceback.extract_stack()
        interesting = []
        for frame in frames:
            if frame.name in plumbing:
                continue
            if not frame.filename.startswith(_OWN_PACKAGE_DIRS):
                continue
            interesting.append(frame)
        result_frames = interesting[-2:]
        if not result_frames:
            return ""
        parts = []
        for f in result_frames:
            filename = os.path.basename(f.filename)
            parts.append(f"{filename}:{f.lineno} {f.name}")
        return " → ".join(parts)

    @staticmethod
    def _assert_unique_section_names(sections: dict) -> None:
        """on_section_updated routes incoming notifications by generator.section_name.
        If two generators share that display name, _find_section_by_name returns the
        first match and the second section's updates get mis-attributed (its baseline
        never refreshes) — so fail fast at startup rather than silently misroute deltas.
        """
        names = [getattr(gen, "section_name", None) for gen in sections.values()]
        if len(set(names)) != len(names):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise RuntimeError(
                f"Duplicate section_name(s) across PromptSectionGenerators: {duplicates}. "
                "Each generator's section_name must be unique."
            )

    def _find_section_by_name(self, section_name: str) -> tuple[str, "PromptSectionGenerator"] | tuple[None, None]:
        """Resolve a generator's display name (e.g. "Weather", "CalendarData") to the
        (placeholder_key, generator) pair. Generators identify themselves via the
        display name passed to PromptSectionGenerator.__init__, but self.sections is
        keyed by the lowercase placeholder slug used in res/prompt.md — so we have to
        translate. Returns (None, None) if no match (e.g. a generator that fired late
        for a section we no longer track)."""
        for key, gen in self.sections.items():
            if getattr(gen, "section_name", None) == section_name:
                return key, gen
        return None, None

    def on_section_updated(self, section_name: str):
        """
        Callback method called by PromptSectionGenerators when they update.

        For each update we choose between two paths to keep the LLM's prompt cache warm:
          - Small diff against the section's LAST-SEEN state (the +side of the most
            recent note, or the baseline if untouched) and we're still inside the 1h
            window → append a transient "section updated" note to MessageHistory and
            keep the prompt bytes unchanged. Notes accumulate as an append-only
            changelog: each is stamped at the moment its change happened and diffs
            only against the immediately preceding state, so a later change to the
            same section adds a fresh note instead of re-stating earlier changes.
          - Large diff, missing baseline, or expired window → full rebuild (which
            re-snapshots all baselines, resets last-seen state, and drops the
            now-obsolete transient notes).

        The state read + diff decision + transient emission all happen under
        _prompt_lock so we can't race with a parallel _assemble_prompt and end up
        diffing against state that's about to be invalidated, or appending a delta
        whose content is already in a freshly-rebuilt prompt.
        """
        if self.test_suite and "history" in self.sections:
            return

        # Translate display name → placeholder key (see _find_section_by_name).
        # During tests we sometimes assign DummySections lacking .section_name and
        # invoke this with the placeholder key directly; fall back to that.
        with self._prompt_lock:
            key, section = self._find_section_by_name(section_name)
            if section is None and section_name in self.sections:
                key, section = section_name, self.sections[section_name]
            if section is None:
                return
            from assistant.commands import Toolkits
            if key in Toolkits.CARRIED_SECTIONS:
                pre_heat_messages = self._on_carried_section_updated_locked(section_name, key, section.get_content())
                if pre_heat_messages is not None:
                    trace(f"Pre-heating after delta note — section={section_name}")
                    self._fire_prompt_callbacks(pre_heat_messages)
                return
            if key not in self._section_positions and self._section_positions:
                # Not in the template (its data reaches the LLM another way, e.g. rendered
                # into a toolkit on load): nothing to diff the prompt against.
                trace(f"Section '{section_name}' has no placeholder in the prompt template — no delta note")
                return

            baseline = self._section_baselines.get(key)
            # Diff against what the LLM was last told, not the frozen baseline:
            # the +side of the most recent note for this section, or the baseline
            # if it hasn't changed since the last rebuild.
            prev = self._section_current.get(key, baseline)
            new_content = section.get_content()
            window_expired = time.time() - self._last_assembly_time >= self._FULL_REBUILD_INTERVAL_S

            do_full_rebuild = False
            pre_heat_messages = None

            if baseline is None or window_expired:
                if baseline is not None:
                    info(f"Full prompt rebuild — section={section_name} reason={self._FULL_REBUILD_INTERVAL_S // 3600}h window expired")
                do_full_rebuild = True
            elif new_content == prev:
                # Nothing changed since the last state the LLM saw — no-op. (A revert
                # baseline→A→baseline is just another honest A→baseline delta below.)
                return
            else:
                # Incremental diff against the previous state. Each note is an
                # independent, append-only changelog entry — we do NOT clear earlier
                # notes for this section; they describe real changes the LLM should
                # keep seeing, stamped at the time they happened.
                diff = self._compute_section_delta(prev, new_content)
                if 0 < len(diff) <= self._DIFF_SIZE_THRESHOLD:
                    self._emit_transient_section_update_locked(section_name, key, diff)
                    # Advance last-seen state so the next change diffs against this one.
                    self._section_current[key] = new_content
                    # The note extends the request prefix, so pre-heat it too — the
                    # prefix before the note is already cached, making this cheap.
                    pre_heat_messages = self._resolved_messages_locked()
                else:
                    info(f"Full prompt rebuild — section={section_name} reason=diff too large ({len(diff)} chars)")
                    # trace(f"Full prompt rebuild diff — section={section_name}:\n{diff}")
                    do_full_rebuild = True

        if pre_heat_messages is not None:
            trace(f"Pre-heating after delta note — section={section_name}")
            self._fire_prompt_callbacks(pre_heat_messages)
            return

        # _assemble_prompt acquires _prompt_lock itself, so call it outside our with block.
        if do_full_rebuild:
            self._assemble_prompt(section_name, force=self.test_suite and "history" not in self.sections)

    @staticmethod
    def _compute_section_delta(old: str, new: str) -> str:
        """Minimal +/- diff between baseline and new content. Empty if no textual diff.

        Strips the file-header lines ("--- " / "+++ ") and the hunk headers ("@@ ... @@"),
        and uses n=0 so no context lines are emitted — the LLM only needs to see what
        was removed and what was added. Shorter output also means more updates fit under
        _DIFF_SIZE_THRESHOLD, so we stay on the delta path longer."""
        diff_lines = list(difflib.unified_diff(
            old.splitlines(), new.splitlines(),
            n=0, lineterm="",
        ))
        # Slice off the two file-header lines rather than filtering by prefix — actual
        # content lines starting with "--" or "++" pick up the diff's leading "-"/"+",
        # becoming "---foo" / "+++bar", which a prefix filter would incorrectly drop.
        if len(diff_lines) >= 2 and diff_lines[0].startswith("--- ") and diff_lines[1].startswith("+++ "):
            diff_lines = diff_lines[2:]
        # Drop hunk headers ("@@ -54,3 +54,3 @@"). Content lines starting with "@@"
        # would appear as "+@@..." / "-@@..." in the diff (they have the +/- prefix)
        # so this strict startswith check leaves real content untouched.
        diff_lines = [line for line in diff_lines if not line.startswith("@@")]
        return "\n".join(diff_lines)

    def _on_carried_section_updated_locked(self, section_name: str, key: str, new_content: str) -> list | None:
        """A section a toolkit carries is never in the prompt: its baseline is what the last
        `load_toolkit` sent (carried_data) and a note goes out only while that toolkit is in
        the LLM's window — any window, a voice preload's row included. A change too big for one note, or one that would push the section's
        notes past _CARRIED_NOTES_BUDGET, becomes a single "load it again" pointer and ends the
        tracking until the next load. Caller holds _prompt_lock; returns the messages to
        pre-heat after a note, else None."""
        from assistant.commands import Toolkits
        toolkit = Toolkits.CARRIED_SECTIONS[key]
        heading = Toolkits.SECTION_HEADINGS.get(key, section_name)
        prev = self._section_current.get(key)
        if prev is None:
            # No baseline: nothing was sent since this process started (the row in the window
            # is from before a restart), or a pointer note ended the tracking. Either way the
            # LLM's copy is stale while the reference is in its window, so one pointer note —
            # not one per change — tells it to load again.
            if toolkit not in self.message_history.loaded_toolkits(window=None):
                trace(f"Section '{section_name}' changed but was never sent (no load_toolkit yet) — no delta note")
                return None
            if key in self._carried_pointer_out:
                trace(f"Section '{section_name}' changed, its pointer note is already out — nothing to add")
                return None
            info(f"Section '{section_name}' changed with the {toolkit} toolkit in the window but no baseline "
                 f"(sent before a restart?) — one pointer note tells the LLM to load it again")
            self._add_carried_pointer_locked(key, heading, toolkit)
            return self._resolved_messages_locked()
        if new_content == prev:
            return None
        if toolkit not in self.message_history.loaded_toolkits(window=None):
            info(f"Section '{section_name}' changed but the {toolkit} toolkit left every window — tracking stops")
            self._section_current.pop(key)
            return None
        diff = self._compute_section_delta(prev, new_content)
        if not diff:
            return None
        accumulated = self._carried_note_chars.get(key, 0)
        if len(diff) <= self._DIFF_SIZE_THRESHOLD and accumulated + len(diff) <= self._CARRIED_NOTES_BUDGET:
            trace(f"Adding section delta: {section_name} ({len(diff)} chars, {accumulated} chars of notes before it)")
            self._add_transient_note_locked(key, f"{heading} updated:\n{diff}")
            self._carried_note_chars[key] = accumulated + len(diff)
            self._section_current[key] = new_content
        else:
            info(f"Section '{section_name}' changed too much for notes ({len(diff)} chars on top of {accumulated}) "
                 f"— one pointer note follows them, tracking stops until the next load_toolkit({toolkit})")
            self._add_carried_pointer_locked(key, heading, toolkit)
            self._section_current.pop(key)
        return self._resolved_messages_locked()

    def _add_carried_pointer_locked(self, key: str, heading: str, toolkit: str):
        self._add_transient_note_locked(key, self._reload_pointer(heading, toolkit))
        self._carried_pointer_out.add(key)

    @staticmethod
    def _reload_pointer(heading: str, toolkit: str) -> str:
        return (f"{heading} has changed since it was sent: run `load_toolkit({toolkit})` for the current state.")

    def _emit_transient_section_update_locked(self, section_name: str, key: str, diff: str):
        """Append an in-memory-only note describing a small section change.

        Caller MUST hold self._prompt_lock — keeps the baseline-read / decision /
        emit sequence atomic with respect to full rebuilds.

        `section_name` is the human-facing display name shown in the note body;
        `key` is the stable placeholder slug used to tag the entry (so all of a
        section's accumulated notes can be dropped together on a full rebuild).

        Rendered to the LLM as a user-role turn with a "System (HH:MM): ..." prefix
        (MessageHistory.render_messages collapses role="system" entries that way);
        we phrase the body so the final string reads naturally, e.g.
        "System (10:30): Prompt section `weather` updated:\\n@@ ..."
        """
        trace(f"Adding section delta: {section_name} ({len(diff)} chars)")
        self._add_transient_note_locked(key, f"Prompt section `{section_name}` updated:\n{diff}")

    def _add_transient_note_locked(self, key: str, content: str):
        self.message_history.add_transient_entry(
            role="system",
            content=content,
            timestamp=time.time(),
            source=MessageHistory.SOURCE_PROMPT_DELTA,
            section=key,
        )

    def is_night_mode_active(self) -> bool:
        """Whether home automation currently reports Night mode (the live snapshot).

        Callers (the autonomous-action night guards) must NOT read self.prompt for
        this: when Night flips via a small section delta the system-prompt bytes are
        intentionally frozen (only a transient note is emitted), so self.prompt lags
        the real state by up to the 1h rebuild interval. The home section's live
        snapshot reflects the flip immediately.
        """
        section = self.sections.get('home')
        if section is None:
            return False
        try:
            return section.is_night()
        except Exception as e:
            trace(f"Night-mode check failed, assuming not Night: {e}")
            return False

    @staticmethod
    def current_time_note(at: float | None = None) -> str:
        """The "(Current time: HH:MM, Weekday, YYYY-MM-DD)" clock note, for now or for `at`.

        Single source of truth for the format. Appended to the trailing user turn so
        the LLM always has a clock without rewriting the cached system prompt bytes.
        Shared by get_messages() and the autonomous-action check, which builds its own
        message list and must inject the same clock — AUTONOMOUS_ACTION_PROMPT asks the
        model to compare elapsed time against the current time.
        """
        now = datetime.fromtimestamp(at) if at is not None else datetime.now()
        return (
            f"(Current time: {now.strftime('%H:%M')}, "
            f"{now.strftime('%A')}, {now.strftime('%Y-%m-%d')})"
        )

    def on_prompt_assembled(self, callback):
        """Register a callback fired whenever the byte prefix of the next request
        changes — after a full prompt rebuild AND after a delta note is appended —
        so KV-cache pre-heating can cover both paths.
        Callback signature: callback(messages: list[dict])"""
        self._on_prompt_assembled_callbacks.append(callback)

    def _resolved_messages_locked(self, window: str | None = None) -> list[dict]:
        """The byte prefix every real request of `window` shares: system prompt with
        per-request placeholders resolved, plus the live history window (including
        transient delta notes; the last request's window when None). get_messages() adds
        only the per-request tail on top of this (time note, thinking instruction,
        incoming query), so pre-heating these exact bytes warms the KV prefix real
        requests reuse. Caller must hold _prompt_lock."""
        window = window or self._last_window
        messages = [dict(m) for m in self._messages]

        if TimeOfDay.get_current() == TimeOfDay.DAY:
            whisper = ""
        else:
            whisper = "It is late or very early, so whisper. Keep responses extra short."

        if messages and messages[0].get("role") == "system":
            content = messages[0]["content"]
            content = content.replace("[THINKING]", "")
            content = content.replace("[WHISPER]", whisper)
            messages[0] = {"role": "system", "content": content}

        if not self.test_suite:
            history = self.message_history.get_history_messages(window=window)
            if self._restate_trimmed_sections_locked():
                history = self.message_history.get_history_messages(window=window)
            messages.extend(history)
        return messages

    def _restate_trimmed_sections_locked(self) -> bool:
        """Put one fresh note back for every section whose delta notes a trim just evicted.

        The system prompt still carries the section's baseline, and the notes were the only
        thing telling the LLM it moved on since — on 2026-10-05 a burst of browser output
        evicted them and the model reported the car still driving, 40 minutes after it
        parked. One note diffing the baseline against the last state the LLM was told
        about, stamped now so it sits inside the kept window, restores the chain without
        touching the cached system prompt. A carried section has no baseline in the prompt;
        its notes have their own pointer fallback, so it is left alone here. Caller holds
        _prompt_lock; returns whether a note was added."""
        dropped = self.message_history.dropped_delta_sections
        if not dropped:
            return False
        self.message_history.dropped_delta_sections = set()
        from assistant.commands import Toolkits
        added = False
        for key in sorted(dropped):
            if key in Toolkits.CARRIED_SECTIONS:
                trace(f"Section '{key}' lost its notes to a trim — carried, left to its pointer")
                continue
            baseline = self._section_baselines.get(key)
            current = self._section_current.get(key)
            if baseline is None or current is None:
                trace(f"Section '{key}' lost its notes to a trim but has no baseline — nothing to restate")
                continue
            diff = self._compute_section_delta(baseline, current)
            if not diff:
                trace(f"Section '{key}' lost its notes to a trim but is back at its baseline — nothing to restate")
                continue
            section = self.sections.get(key)
            name = getattr(section, "section_name", "") or key
            info(f"Section '{name}' lost its notes to a trim — one fresh note restates its current state")
            self._emit_transient_section_update_locked(name, key, diff)
            added = True
        return added

    def _fire_prompt_callbacks(self, messages: list[dict]):
        """Invoke the registered pre-heat callbacks. Never called under _prompt_lock —
        callbacks may do slow work (network requests to the LLM backend)."""
        for callback in self._on_prompt_assembled_callbacks:
            try:
                callback(messages)
            except Exception as e:
                trace(f"Post-assembly callback failed (non-critical): {e}")

    def _get_photo_count(self) -> int:
        """The {PHOTO_COUNT} value, fetched at server start and refreshed at most
        daily. The live count creeps up continuously as photos sync; re-reading it
        on every prompt rebuild would change the cached prompt bytes for a number
        nobody needs to be exact (it's rendered with a ~ prefix). On a failed
        refresh, keep serving the stale count rather than flapping the bytes.
        """
        age = time.time() - self._photo_count_fetched_at
        if self._photo_count is not None and age < self._PHOTO_COUNT_REFRESH_S:
            return self._photo_count
        try:
            self._photo_count = FloxyDatabase.get_instance().get_photo_count()
            self._photo_count_fetched_at = time.time()
            trace(f"Photo count refreshed: ~{self._photo_count} (next refresh in 24h)")
        except Exception as e:
            if self._photo_count is None:
                raise
            trace(f"Photo count refresh failed, keeping stale value {self._photo_count}: {e}")
        return self._photo_count

    def _assemble_prompt(self, section_name: str = "", *, force: bool = False):
        """Assemble the complete prompt from all sections."""
        # In test suite mode, we use a static prompt and never reassemble
        if self.test_suite and not force:
            return

        trigger_trace = self._get_trigger_trace()
        fine(f"Prompt regeneration because of {section_name} ({trigger_trace})")

        with self._prompt_lock:
            # Load base template
            with open("res/prompt.md", "r") as file:
                prompt = file.read()

            # Snapshot baselines from the SAME get_content() calls the substitution uses,
            # so on_section_updated diffs against the exact bytes that landed in self.prompt.
            # A carried section (Toolkits.CARRIED_SECTIONS) is not in the prompt: its baseline
            # is what the last load_toolkit sent, and it survives the rebuild with its notes.
            from assistant.commands import Toolkits
            new_baselines = {key: gen.get_content() for key, gen in self.sections.items()
                             if key not in Toolkits.CARRIED_SECTIONS}
            self.prompt = self._substitute(prompt, new_baselines)
            self._messages = [{"role": "system", "content": self.prompt}]

            # Re-baseline for the next round of delta decisions, under _prompt_lock so a
            # racing delta thread cannot read a stale baseline. The notes already in the
            # window stay there: the new prompt holds their content, but they are rows the
            # LLM was sent, and dropping them would change the window's bytes from the first
            # one on — the next request then rewrote the whole window (prod, 2026-10-10..12:
            # a 12–22K-token write after each of the 13–93 rebuilds a day). They read as a
            # dated log of what changed and leave with the trim like any other row.
            self._section_baselines = new_baselines
            # Reset last-seen state to the freshly-baked bytes: the new prompt now
            # contains everything, so the next delta diffs against the baseline again. A
            # carried section keeps its last-sent state.
            carried = {key: content for key, content in self._section_current.items()
                       if key in Toolkits.CARRIED_SECTIONS}
            self._section_current = dict(new_baselines) | carried

            # Update last assembly time
            self._last_assembly_time = time.time()
            messages_copy = self._resolved_messages_locked()

        self._fire_prompt_callbacks(messages_copy)

    _TEST_SUITE_PHOTO_COUNT = "~43034"

    def _substitute(self, text: str, sections: dict[str, str], photo_count: str | None = None) -> str:
        """Fill every placeholder the templates and toolkit files use: the sections given — by
        their own placeholder, or in the {INTEGRATION_DATA} block for the ones with a heading —
        then {TOOLKITS} (assistant.commands.Toolkits, the toolkits offered now), {DESTINATIONS}
        (assistant.Destinations, where post_message can reach), {PHOTO_COUNT}
        (`photo_count` if given, else the DB's, refreshed at most daily), {WAKE_WORD},
        {WAKE_INTERVAL_MINUTES} and {EMOTION_TAGS} (config-driven)."""
        from assistant.commands import Toolkits
        for key, content in sections.items():
            text = text.replace("{" + key.upper() + "}", content)
        if self.DATA_BLOCK in text:
            text = text.replace(self.DATA_BLOCK, self._render_data_block(sections))
        if "{TOOLKITS}" in text:
            text = text.replace("{TOOLKITS}", Toolkits.render_index(self._monitor()))
        if "{DESTINATIONS}" in text:
            text = text.replace("{DESTINATIONS}", Destinations.describe())
        if "{PHOTO_COUNT}" in text:
            text = text.replace("{PHOTO_COUNT}", photo_count or f"~{self._get_photo_count()}")
        if self.EMOTION_TAGS in text:
            text = re.sub(re.escape(self.EMOTION_TAGS) + r"\n*", lambda _: self._emotion_tags_block(), text)
        text = text.replace("{WAKE_WORD}", config.get('wake_word', 'Michelle'))
        return text.replace("{WAKE_INTERVAL_MINUTES}", self._wake_interval_minutes_str())

    EMOTION_TAGS = "{EMOTION_TAGS}"
    EMOTION_TAGS_FILE = RES_DIR / "voice_emotions.md"

    def _emotion_tags_block(self) -> str:
        """The {EMOTION_TAGS} text of the voice toolkit: the audio tag reference while
        `tts.emotions` is on, nothing while it is off (the synthesizers strip tags then)."""
        if not (config.get('tts', {}) or {}).get('emotions', False):
            trace("tts.emotions is off — the voice toolkit goes without the emotion tags block.")
            return ""
        trace("tts.emotions is on — the voice toolkit carries the emotion tags block.")
        return self.EMOTION_TAGS_FILE.read_text(encoding="utf-8").rstrip() + "\n\n"

    def _render_data_block(self, contents: dict[str, str]) -> str:
        """The {INTEGRATION_DATA} text: every running section with a heading, in SECTION_ORDER,
        as heading, intro and data; one with nothing to say is left out heading and all. Caller
        holds _prompt_lock (the sections change under it)."""
        parts, left_out = [], []
        for key, gen in self.sections.items():
            if not gen.heading or key not in contents:
                continue
            content = contents[key].strip()
            if not content:
                left_out.append(key)
                continue
            parts.append("\n".join(part for part in (gen.heading, gen.intro.strip(), content) if part) + "\n")
        if left_out:
            trace(f"Data block: {', '.join(left_out)} have nothing to show — left out, heading and all")
        return "\n".join(parts)

    def render_template(self, text: str) -> str:
        """`text` with the placeholders filled from the sections' current content — what a
        toolkit reference goes through when load_toolkit renders it."""
        with self._prompt_lock:
            sections = {key: gen.get_content() for key, gen in self.sections.items()}
        return self._substitute(text, sections)

    def carried_data(self, toolkit, record: bool = True) -> str:
        """The toolkit's carried sections (Toolkits.CARRIED_SECTIONS) rendered as load_toolkit
        sends them after the reference, current as of now; empty for a toolkit that carries
        none. What goes out becomes the section's last-seen state, so later changes reach the
        LLM as delta notes while the toolkit stays in its window; the notes of the previous
        send stay in the window (this text supersedes them, and the window is append-only) but
        the notes budget and the pointer start over. `record=False` renders without any of
        that — for a send that never reaches the shared history (the autonomous check's own
        messages), which must not pass for the baseline the history's readers were sent."""
        from assistant.commands import Toolkits
        if not toolkit.sections:
            return ""
        with self._prompt_lock:
            contents = {key: self.sections[key].get_content() for key in toolkit.sections if key in self.sections}
            if record:
                for key, content in contents.items():
                    self._section_current[key] = content
                    self._carried_note_chars[key] = 0
                    self._carried_pointer_out.discard(key)
        data = Toolkits.render_sections(toolkit, contents)
        info(f"load_toolkit: {toolkit.name} brings its {', '.join(contents) or 'missing'} data ({len(data)} chars"
             f"{'' if record else ', not recorded as the baseline'})")
        return data

    def untracked_sections(self, toolkit) -> list[str]:
        """The toolkit's carried sections whose last-sent state is unknown: nothing was sent
        since this process started (a restart with the reference still in the window) or a
        pointer note ended the tracking. A toolkit in the window with one of these needs its
        data sent again before the LLM can be trusted with it."""
        if not toolkit.sections:
            return []
        with self._prompt_lock:
            return [key for key in toolkit.sections if key in self.sections and key not in self._section_current]

    def get_messages(self, extended_thinking: bool = False, time_note: str | None = None,
                     window: str = WINDOW_TEXT) -> list[dict]:
        """Get the current messages (for LLM) with the request channel's history `window`
        (ChannelProfile.window). `time_note` replaces the clock note computed here: a
        byte-prefix request passes the one it will persist with the exchange, so the note
        the LLM saw and the row the next request renders are the same bytes."""
        with self._prompt_lock:
            if window != self._last_window:
                trace(f"PromptManager: the {window} history window is served now, the pre-heats follow it")
                self._last_window = window
            # Shared byte prefix (system prompt + live history) — same bytes the
            # pre-heat callbacks warm, so the KV cache prefix is fully reusable.
            # The test suite uses a fixed fixture instead of the live, mutable history,
            # but the turns are rendered identically (MessageHistory.render_messages),
            # so the LLM receives the same message structure either way.
            messages = self._resolved_messages_locked(window)
            if self.test_suite:
                messages.extend(self._test_history_messages)
                messages.extend(self._test_toolkit_messages)

            # Inject current time as a user-role note attached to the trailing turn.
            # This used to be a `{TIME_DATE}` section in the system prompt, which
            # rewrote the prompt bytes every minute and invalidated the 5-min
            # message-level KV cache on every clock tick. Putting it on the last
            # user turn (which is never cached — DialogManager.stream_llm_response
            # merges the incoming query into this same turn) costs zero cache hits.
            # Skipped in test_suite so unit tests don't depend on wall-clock time.
            if not self.test_suite:
                MessageHistory.append_user_turn(messages, time_note or self.current_time_note())

            # Append the thinking instruction as a user message. This keeps the
            # system message identical to the pre-heat, so the KV cache prefix is
            # fully reusable. If history already ends in a user turn (e.g. an
            # unanswered incoming message or a system event), merge into it rather
            # than appending a second consecutive user turn — Gemini and other
            # providers reject same-role adjacency.
            if extended_thinking:
                trace("Using extended thinking mode.")
                MessageHistory.append_user_turn(messages, THINKING_TAG_INSTRUCTION)

            return messages

    def get_section_stats(self) -> list[tuple[str, int, float]]:
        """
        Get statistics for all sections for TUI visualization.
        Returns list of tuples: (section_name, size_bytes, last_update_time)

        A section without a placeholder in the template (one a toolkit carries, see
        Toolkits.CARRIED_SECTIONS) is not in the prompt: it gets no bar, and its bytes are
        not subtracted from the static size. They show in the history bar once loaded.
        """
        stats = []
        total_dynamic_size = 0

        in_prompt = [key for key in self.sections if key in self._section_positions or not self._section_positions]
        ordered_keys = sorted(in_prompt, key=lambda k: (self._section_positions.get(k, float('inf')), k))

        for section_key in ordered_keys:
            section_generator = self.sections[section_key]
            content = section_generator.get_content()
            size_bytes = len(content.encode('utf-8'))
            last_update = section_generator.last_update_time
            stats.append((section_key, size_bytes, last_update))
            total_dynamic_size += size_bytes

        # History is sent as message items, not in the prompt — report it separately
        # (and exclude it from the static-size math below). Size the actual payload
        # (rendered message turns) rather than the console text rendering, so the bar
        # reflects the bytes sent to the LLM.
        history_bytes = sum(
            len(msg["content"].encode('utf-8'))
            for msg in self.message_history.get_history_messages()
        )
        stats.append(('history', history_bytes, self.message_history.last_update_time))

        # Calculate static portion (total prompt - all dynamic sections).
        total_prompt_size = len(self.prompt.encode('utf-8'))
        static_size = total_prompt_size - total_dynamic_size

        stats.append(('static', static_size, 0.0))

        return stats

    def load_test_suite_history(self, path: str):
        """Load the prompt test suite's fixed history fixture (see get_messages). Like
        testsuite_prompt.md it is built from a real household and absent from a public checkout."""
        if not os.path.exists(path):
            info(f"No {path} in this checkout, the test suite runs without a history fixture.")
            return
        self._test_history_messages = MessageHistory.load_fixture_messages(path)
        info(f"Loaded {len(self._test_history_messages)} test-suite history turns from {path}")

    def update_history(self, user_name: str, user_query: str, assistant_reply: str,
                       source: str = None, location: str = None, pre_heat: bool = True,
                       timestamp: float | None = None, time_note: str | None = None):
        """Update conversation history with a user-assistant exchange.

        pre_heat=False for mid-request bookkeeping (command follow-up rounds):
        the follow-up inference starts immediately after and would race the
        pre-heat on the single-slot Ollama server — and the follow-up itself
        warms the same prefix anyway. The request's terminal exchange passes
        True. `timestamp` and `time_note` come from a byte-prefix request
        (MessageHistory.add_exchange)."""
        self.message_history.add_exchange(user_name, user_query, assistant_reply, source, location,
                                          timestamp=timestamp, note=time_note)

        # The rendered exchange extends the request prefix (the assistant's own
        # generated tokens are NOT reusable — history is re-rendered with turn
        # wrappers and timestamps, so the bytes differ). Warm it now, while the
        # runner is idle right after the turn, instead of at the next query.
        if not pre_heat:
            trace("Skipping pre-heat: mid-request history update (follow-up round pending).")
            return
        if not self._on_prompt_assembled_callbacks:
            return
        with self._prompt_lock:
            messages = self._resolved_messages_locked()
        self._fire_prompt_callbacks(messages)

    def add_command_result(self, result: str, cap: bool = True, source: str | None = None):
        """Persist a command round's result so later turns still see it (no pre-heat:
        the follow-up round that consumes it starts right away). `cap=False` for a
        toolkit reference, which is stored whole; `source` stamps a preloaded one with
        its window (MessageHistory.preload_source)."""
        self.message_history.add_command_result(result, cap=cap, source=source)

    @staticmethod
    def save_prompt_to_file(messages: list[dict]):
        """Save the full messages array sent to the LLM to log/prompts directory."""
        from common.logging import disable_file_logging, masked_for_log
        if disable_file_logging:
            return
        try:
            os.makedirs("log/prompts", exist_ok=True)
            unique_id = str(uuid.uuid4())
            prompt_file = f"log/prompts/{unique_id}.md"

            # The history holds real values (redaction happens at the LLM boundary), so the
            # dump is masked like every other log artefact.
            formatted = masked_for_log(PromptManager._format_messages(messages))

            with open(prompt_file, "w") as f:
                f.write(formatted)

            prompt_file2 = "log/current_prompt.md"
            with open(prompt_file2, "w") as f:
                f.write(formatted)

            trace(f"Prompt saved to {prompt_file} ({len(formatted)} bytes)")
        except Exception as e:
            trace(f"Error during prompt save: {e}")

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        """Format the full messages list into a readable log file."""
        parts = []
        image_index = 0
        prompt_id = str(uuid.uuid4())[:8]
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block["text"])
                        elif block.get("type") == "image":
                            path = PromptManager._save_image_block(block, prompt_id, image_index)
                            image_index += 1
                            text_parts.append(f"[IMAGE: {path}]" if path else "[IMAGE]")
                content = "\n".join(text_parts)
            parts.append(f"# [{role}]\n\n{content}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _save_image_block(block: dict, prompt_id: str, index: int) -> str | None:
        from common.logging import disable_file_logging
        if disable_file_logging:
            return None
        try:
            source = block.get("source", block)
            data_b64 = source.get("data", "")
            media_type = source.get("media_type", "image/jpeg")
            ext = "png" if "png" in media_type else "jpg"

            os.makedirs("log/images", exist_ok=True)
            filename = f"prompt_{prompt_id}_{index}.{ext}"
            path = f"log/images/{filename}"

            with open(path, "wb") as f:
                f.write(base64.b64decode(data_b64))
            return path
        except Exception:
            return None

    def stop(self):
        """Stop all background threads and clean up resources."""
        if not self.test_suite:
            self._monitor().remove_listener(self._on_integrations_changed)
        for name, section in list(self.sections.items()):
            if hasattr(section, 'stop'):
                section.stop()

