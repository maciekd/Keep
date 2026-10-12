import re
import shutil
import time
import uuid
import difflib
import threading
from datetime import datetime, timedelta, date
from pathlib import Path

from .. import Destinations, MemoryFile, Procedures, WatchList
from ..llm import create_llm, parse_llm_config
from common.ConfigManager import ConfigManager
from common.FloxyDatabase import FloxyDatabase, NO_INTERACTIONS
from assistant.integration.calendar.CalendarAccess import CalendarAccess, EVENTS_UNAVAILABLE, NOT_CONFIGURED
from assistant.integration.calendar.Event import NO_EVENTS
from common.logging import info, error, trace
from ..llm.AbstractLLM import LLMRefusalError
from common.Paths import MEMORY_DIR, MEMORY_LOG_DIR
from common.vault.Redactor import redact_for_storage

HISTORY_SOURCE = "memory_curation"   # the `source` of the system row a finished curation leaves


# A span longer than this is curated in chunks of this many days, one after another, each
# reading the bank the previous one wrote (`run_span`).
CURATION_CHUNK_DAYS = 4
# The size the curator is told to keep the bank under. The LLM cannot count words in a 40 KB
# document, so the prompt also states the measured count ({MEMORY_SIZE}) rather than trusting it to.
MEMORY_WORD_LIMIT = 4000
# How long a nightly run waits for its batch before it makes the ordinary full-price call
# (`ai.memory_curator.batch_wait_minutes`): from the 04:00 run, done before the house wakes.
BATCH_WAIT_MINUTES = 90
# What the curator may do to the watch list and the procedures, in the command block after the
# bank. It may add to a procedure but never replace or delete one: a procedure is rewritten
# only by the assistant that has the working flow in front of it, or by the user.
# Not `checked_watch`: only a real check may stamp an item as looked at.
CURATOR_COMMANDS = ("save_watch", "delete_watch", "save_procedure", "procedure_note")
# The one message the curator may send: a tip for other households' Keeps, posted to the group
# `ai.memory_curator.tips_destination` names (a post_message destination). Unset, the command
# is skipped like any other that is not the curator's. The destination the model writes is
# ignored in favour of the configured one, so a run can never post anywhere else.
TIPS_COMMAND = "post_message"
TIPS_KEY = 'ai.memory_curator.tips_destination'

PROMPT = """
You are a long-term memory curator for a state of the art home automation AI system serving a family. Your role is to maintain a
comprehensive memory bank about the family and household that enables personalized automation.

You are called periodically — every night once the system has been running for a while, more often right after a
fresh install while the memory bank is still being built up — to analyze everything that happened in a period and
update the long-term memory. This run covers {PERIOD}. Today is {TODAY}.

The long-term memory has three parts:
- the **memory bank**: what is durably true of the family — people, preferences, relationships, contacts, standing
  rules — plus a short "Ongoing projects" section: where the family's own projects, to-dos and open decisions stand
  (a renovation, a purchase being considered). It is in front of the AI in every conversation. You rewrite it.
- the **watch list**: what the AI is expected to act on later — notify, remind, check. A delivery or order it is
  following, an invoice it was asked to watch, a reminder someone asked for, a task of its own. NOT an appointment the
  calendar already holds (unless the AI has a job attached to it, such as a reminder to cancel in time), NOT the
  family's own to-dos, plans and open decisions that it merely heard about, and nothing that ends today. The test: would
  the AI ever speak up about this unprompted? If not, it is not a watch item. Expect a handful of items, not dozens.
  One item per thing, each with a note that holds its current state. The AI keeps it up to date during the day; you
  tidy up what it missed.
- the **procedures**: how something is done — login flows, checkouts, bookings, readings. The AI loads one when it
  needs it, so a procedure can be as detailed as it has to be. You only see their index.

{STT_NOTE}

## Recent interactions:
{INTERACTION_HISTORY}

## Current memory bank:
{CURRENT_MEMORY}

{MEMORY_SIZE}

## Current watch list (open items):
{WATCHLIST}

## Procedures (index only):
{PROCEDURES}

## Your task:
Update the memory bank based on new information. You may add, modify, remove and restructure information as needed.
Keep the memory bank concise - no more than {WORD_LIMIT} words. Condense or remove outdated/less important information as needed.

Then bring the watch list in line with what the period shows, with commands (see the output format).

Rules:
- The bank does NOT include command names or action IDs (the AI knows what commands it can run)
- The bank does NOT include command syntax or how to interpret commands (the AI already understands natural language and has extensive knowledge of available commands). A procedure is different: there the exact commands and code that worked are the point
- Do NOT include song/playlist/favorite IDs
- Focus on personal preferences, patterns, and facts about the family that help personalization
- Make sure to keep the entire memory under {WORD_LIMIT} words
- When a user corrects a misheard command (e.g. by saying "No, I said X" right after), the original action was based on a MISHEARD phrase. Record the STT correction, but do NOT add anything from the misinterpreted action as a genuine preference
- Be very mindful of making assumptions - errors sneak easily into the memory bank and can propagate over time - so only enter things you are sure of in the memory
- Remove a fact only when it of extremely low value or recent interactions give positive evidence against it: contradicted, superseded, explicitly retracted, or expired. Not being mentioned in this period's logs is not necessarily staleness
- This memory bank is the system's ONLY long-term storage — anything dropped is permanently forgotten. Be mindful of deleting valuable information
- The bank deliberately holds operational data the system acts with: credentials, payment details. Preserve these verbatim — never redact, mask or drop them
- Do not assume things, be very mindful of what you put into the memory. Only facts you are sure of.
- The bank holds no watch items: something the AI must act on later (see the watch list's definition above) that is
  in the bank or in the period's interactions and still open today belongs on the watch list — `save_watch(...)` it
  (under the name of the open item that already covers it, when there is one) and leave it out of the bank. One that
  is already over is simply dropped. What an item leaves behind that is durably true (a new possession, a contact, a
  preference learnt) does go in the bank
- The family's own projects, to-dos and open decisions that the AI only heard about go in the bank's "Ongoing
  projects" section, a line or two per project, kept current and freely rewritten: drop what is done, never let it
  grow into a log. Appointments already in the calendar and plans for a single day go nowhere
- The bank holds no how-tos: a multi-step way of doing something belongs in a procedure. When the bank or the period
  holds one that no procedure in the index covers, save it with `save_procedure(...)`, complete and verbatim (every
  step, URL, endpoint, code snippet, warning), and leave at most a one-line pointer in the bank. You cannot replace
  an existing procedure; add what the period taught with `procedure_note(...)`
- Watch list upkeep: delete an item the period shows is done, delivered, paid, cancelled or past
  (`delete_watch`); rewrite the note of one whose state changed (`save_watch` under its existing name — the note
  REPLACES the old one, so carry over every detail that still holds). Leave an item alone when the period says nothing about it, unless it
  is marked STALE and clearly over

Format the memory nicely with markdown. Add sections when an item contains more information that fits in a sentence or two.

## Output format
The COMPLETE updated memory bank (markdown) between <memory> and </memory>, then the commands, one per line, between
<commands> and </commands> (leave it empty when there is nothing to do). Nothing else. Every command starts on a
line of its own.

The commands:
- `save_watch("name", \"\"\"note\"\"\", "due", "every")` — a new item, or the new note of the item with that name;
  the note in a triple-quoted block; `due` as YYYY-MM-DD, optional; `every` how often the AI must check it ("1d",
  "weekly"), optional — omitted, both are kept as they were. Never `checked_watch`: only a real check stamps it
- `delete_watch("name", "outcome")` — the item is over
- `save_procedure("name", "when to load it, one line", \"\"\"the full text\"\"\")` — a NEW procedure; the text in a
  triple-quoted block
- `procedure_note("name", "the lesson")` — append a lesson to an existing procedure
{TIPS}"""

# Appended to the prompt only while `ai.memory_curator.tips_destination` is set.
TIPS_PROMPT = """
## Tips for other Keeps
Other households run their own Keep, and the Keeps share tips in the Telegram group "{TIPS_DESTINATION}". Go over the
period for what another household's Keep would find useful — something learnt about a site, a service, a device, or
about Keep itself (a command that behaves unexpectedly, a configuration that took work) — and post each one as a short
plain-language tip, in the command block:
`post_message("{TIPS_DESTINATION}", \"\"\"Tip from {OWNER}'s Keep: ...\"\"\")` — the prefix names whose Keep it comes from.
Never the household's people, places, accounts, schedules or anything from the vault: the group is read by other
households. Nothing worth posting is the normal outcome of a run.
"""

# Only worth the LLM's attention when a microphone is actually feeding the assistant.
STT_NOTE = """The AI receives commands from the users using STT tech, the STT accuracy is not perfect and phrases are misheard sometimes. Keep
track of common misheard phrases and their actual meaning - as a way to mitigate the sub-par accuracy of the STT system.
"""

# --- //memory_split: the one-off move of an old, single-file bank into the three parts --------
# Two calls, so neither reply has to carry the whole bank: the first lifts the tracked items
# and the how-tos out as commands, the second rewrites the bank without them.

SPLIT_EXTRACT_PROMPT = """
You maintain the long-term memory of a home automation AI serving a family. Until now that memory was one document,
the memory bank below. It is being split into three parts:
- the **memory bank** keeps what is durably true of the family: people, preferences, relationships, contacts,
  standing rules, how the house is set up — and where the family's own projects, to-dos and open decisions stand.
- the **watch list** takes what the AI is expected to act on later — notify, remind, check. A delivery or order it is
  following, an invoice it was asked to watch, a reminder someone asked for, a task of its own. NOT an appointment the
  calendar already holds (unless the AI has a job attached to it, such as a reminder to cancel in time), NOT the
  family's own to-dos, plans and open decisions that it merely heard about, and nothing that ends today. The test: would
  the AI ever speak up about this unprompted? If not, it is not a watch item. Expect a handful of items, not dozens.
  One item per thing.
- the **procedures** take every how-to: a multi-step way of doing something (a login flow, a checkout, a booking, a
  reading, a cleanup routine), with the lessons learnt about it.

Today is {TODAY}.

## The memory bank:
{BANK}

## Your task:
Lift the watch items and the procedures out of the bank, as commands. Do NOT output the bank.

- `save_watch("name", \"\"\"note\"\"\", "due")` — the note in a triple-quoted block. One per watch item (as defined above) that is still open today; when in
  doubt, it is not one and stays in the bank. The name is a short label
  (`tv-delivery`, `dentist-charlotte`). The same thing is often mentioned
  in several places of the bank (a person's section, a monitoring list, a calendar-like list, a loose note at the end):
  merge them into ONE item whose note holds the current state and every detail worth keeping (order numbers, amounts,
  references, names, phone numbers, times). Where two mentions disagree, the later-dated one is the current state.
  `due` is the date it is due or expected as YYYY-MM-DD; leave it out when there is none. Skip what is already over.
- `save_procedure("name", "when to load it, one line", \"\"\"the full text\"\"\")` — one per how-to, the text in a
  triple-quoted block. Copy every step, URL, endpoint, code snippet, number and warning VERBATIM from the bank; do not
  summarise, do not drop a detail, do not invent a step. Give related how-tos separate procedures (one per site or
  task), and let the name say what it is (`vklass-login`, `online-purchases`).

Output only the commands, each starting on a line of its own, between <commands> and </commands>.
"""

SPLIT_BANK_PROMPT = """
You maintain the long-term memory of a home automation AI serving a family. Until now that memory was one document,
the memory bank below. It has just been split: what the AI must act on later (deliveries and orders it follows,
reminders it was asked for) moved to the watch list, the how-tos moved to the procedures. Both are shown below as
they now exist.

Today is {TODAY}.

## The memory bank (before the split):
{BANK}

## The watch list (now holds these):
{WATCHLIST}

## The procedures (now hold these):
{PROCEDURES}

## Your task:
Output the memory bank as it should be after the split: everything that is durably true of the family — people,
preferences, relationships, contacts, standing rules, how the house is set up, corrections of misheard phrases —
and nothing that now lives in the watch list or in a procedure.

- Remove what the watch list and the procedures hold. A pointer of at most one line may stay where the AI must know
  that something exists before it would think of loading it ("Anything asked 'på Vklass' → the vklass-login procedure").
- The family's own projects, to-dos and open decisions that did NOT become watch items stay in the bank: gather
  them under an "Ongoing projects" section, a line or two per project, each thing said once (the old bank often
  repeats them in a to-do list and again under a person or a room). Drop the ones that are already over.
- Remove appointments the calendar holds and plans for a single day.
- Remove dated day-level chatter (what happened on a given day) unless it carries a lasting fact; keep the fact.
- Keep everything else VERBATIM. Apart from the "Ongoing projects" section this is a move, not a rewrite: do not
  rephrase, condense, merge or reorder lines that stay, and keep credentials, numbers and names exactly as they are.
- Loose notes at the end of the bank that stay are filed under the section they belong to.

Output the complete bank (markdown) between <memory> and </memory>, and nothing else.
"""

"""
At some point a seed memory will need to be authored. It MUST contain

# STT Corrections
# Automation Rules
headlines

 + one headline per person in the family with basic info for each member
"""


class LongTermMemoryCurator:
    """Rewrites the long-term memory bank from the interaction history of a period — nightly
    once the system is settled, more often while it is new — and tidies the watch list and
    adds to the procedures through the command block that follows the bank in its reply."""

    _curation_lock = threading.Lock()

    def __init__(self,
                 floxy_db: FloxyDatabase = None,
                 memory_file_path: str = None,
                 llm=None,
                 test_mode: bool = False,
                 watchlist_dir: Path = None,
                 procedures_dir: Path = None,
                 batch: bool = False):
        """
        Initialize the memory curator.

        Args:
            floxy_db: FloxyDatabase instance (defaults to FloxyDatabase.get_instance())
            memory_file_path: Path to the memory bank (defaults to memory/bank.md)
            llm: LLM instance to use (defaults to creating ClaudeLLM)
            test_mode: If True, skip database initialization (use curate_memory_with_data only)
            watchlist_dir: The watch list the curator reads and tidies (defaults to memory/watchlist)
            procedures_dir: The procedures it indexes and adds to (defaults to memory/procedures)
            batch: Nobody waits for this curation (the nightly run), so it may take the half-price
                batch API and an answer that is up to `ai.memory_curator.batch_wait_minutes` away
        """
        self.batch = batch
        self.watchlist_dir = watchlist_dir
        self.procedures_dir = procedures_dir
        if test_mode:
            self.floxy_db = None
        else:
            self.floxy_db = floxy_db if floxy_db is not None else FloxyDatabase.get_instance()
        self.memory_file_path = memory_file_path if memory_file_path is not None else MemoryFile.MEMORY_FILE

        if llm is None:
            llm_name, model = parse_llm_config('memory_curator')
            # One-shot call: a prompt cache written here is never read again,
            # so caching would only add the cache-write surcharge on ~128k input tokens.
            self.llm = create_llm(llm_name, model, prompt_cache=False)
            self.llm.initialize()
        else:
            self.llm = llm

    @classmethod
    def run(cls, period_start: datetime | date = None, period_end: datetime | date = None,
            curator: 'LongTermMemoryCurator' = None, history=None, batch: bool = False) -> dict | None:
        """Curate the period, save the bank and its debug log. One curation at a time: a run
        that finds another in progress is skipped (the next one covers its period anyway).
        With a `MessageHistory` the completion is recorded as a system row, so it shows in
        the Home tab chat like the other system events and the LLM knows its memory changed.
        `batch` is for a run nobody waits for (see `__init__`). Returns the metadata, or None
        when skipped."""
        if not cls._curation_lock.acquire(blocking=False):
            info("Rem: already dreaming — skipping this run.")
            return None
        try:
            return cls._curate_period(curator or cls(batch=batch), period_start, period_end, history)
        finally:
            cls._curation_lock.release()

    @classmethod
    def run_span(cls, start: date, end: date, history=None, curator: 'LongTermMemoryCurator' = None,
                 on_chunk=None, first_chunk: int = 0) -> dict | None:
        """Curate the days start..end (end inclusive) in `CURATION_CHUNK_DAYS` chunks, in
        order, under one lock: chunk n+1 reads the bank chunk n wrote, and the nightly run
        that lands mid-span is skipped like any concurrent run. A chunk that fails (an
        exception, a truncated reply) or has nothing to curate does not stop the span.
        `on_chunk(index, total, metadata | None)` is called after each chunk, 1-based, so a
        caller can persist where it got to and resume with `first_chunk`; a callback that
        returns False stops the span there (`aborted` in the summary, no history row). A span
        of several chunks leaves one history row when it ends, not one per chunk — the chunk
        completions are a progress bar's business, not the transcript's. Returns a summary
        (`chunks`, `curated`, `empty`, `failed`, `aborted`, `interaction_count`, the span),
        or None when another curation held the lock."""
        if not cls._curation_lock.acquire(blocking=False):
            info("Rem: already dreaming — skipping this span.")
            return None
        try:
            curator = curator or cls()
            periods = cls.chunks(start, end)
            total = len(periods)
            summary = {'chunks': total, 'curated': 0, 'empty': 0, 'failed': 0, 'aborted': False, 'interaction_count': 0,
                       'period_start': periods[0][0].isoformat(timespec='minutes'),
                       'period_end': periods[-1][1].isoformat(timespec='minutes')}
            info(f"Rem: curating {start:%Y-%m-%d} .. {end:%Y-%m-%d} in {total} chunk(s) of up to "
                 f"{CURATION_CHUNK_DAYS} days" + (f", resuming at chunk {first_chunk + 1}" if first_chunk else ""))
            for index, (period_start, planned_end) in enumerate(periods[first_chunk:], start=first_chunk + 1):
                period_end = cls.live_end(planned_end)
                if period_end != planned_end:
                    info(f"Rem: chunk {index}/{total} was planned to end at {planned_end:%H:%M} — it runs now, "
                         f"so it ends at {period_end:%H:%M}")
                    summary['period_end'] = period_end.isoformat(timespec='minutes')
                info(f"Rem: chunk {index}/{total} — {period_start:%Y-%m-%d %H:%M} .. {period_end:%Y-%m-%d %H:%M}")
                try:
                    metadata = cls._curate_period(curator, period_start, period_end, history, record=total == 1)
                except Exception as e:
                    error(f"Rem: chunk {index}/{total} failed: {type(e).__name__}: {e} — carrying on with the next")
                    metadata = None
                    summary['failed'] += 1
                else:
                    if metadata.get('empty'):
                        summary['empty'] += 1
                    elif metadata.get('truncated'):
                        summary['failed'] += 1
                    else:
                        summary['curated'] += 1
                    summary['interaction_count'] += metadata.get('interaction_count', 0)
                if on_chunk is not None and on_chunk(index, total, metadata) is False:
                    summary['aborted'] = True
                    info(f"Rem: span stopped by the caller after chunk {index}/{total}")
                    return summary
            info(f"Rem: span complete — {summary['curated']} of {total} chunk(s) curated, {summary['empty']} empty, "
                 f"{summary['failed']} failed, {summary['interaction_count']} interactions")
            if total > 1 and history is not None:
                history.add_context_message("system", cls.span_note(summary), time.time(), source=HISTORY_SOURCE)
            return summary
        finally:
            cls._curation_lock.release()

    @staticmethod
    def chunks(start: date, end: date, chunk_days: int = CURATION_CHUNK_DAYS,
               now: datetime = None) -> list[tuple[datetime, datetime]]:
        """The day-aligned (period_start, period_end exclusive) pairs that cover start..end
        (dates, end inclusive) in chunks of `chunk_days`; the last is shorter when the span
        is not a multiple. A chunk reaching into the future ends now, so a span that ends today
        covers what has happened so far and nothing that hasn't — `run_span` moves that end to
        the now the chunk actually runs at (`live_end`). Deterministic apart from that clamp,
        so a resumed span gets the same boundaries."""
        now = now or datetime.now()
        end = min(end, now.date())
        if end < start:
            raise ValueError(f"Curation span is empty: {start} .. {end}")
        periods = []
        day = start
        while day <= end:
            chunk_end = min(day + timedelta(days=chunk_days), end + timedelta(days=1))
            period_start = datetime.combine(day, datetime.min.time())
            period_end = min(datetime.combine(chunk_end, datetime.min.time()), now)
            periods.append((period_start, period_end))
            day = chunk_end
        return periods

    @staticmethod
    def live_end(planned_end: datetime, now: datetime = None) -> datetime:
        """The end a chunk gets when it runs: a day-aligned end stays, an end `chunks` cut at
        now (not midnight) is cut again at the now the chunk runs at, up to the midnight it was
        reaching for. A span's chunks run one after another for minutes (the first curation of
        an install took a quarter of an hour), and the last chunk — the one that describes the
        present — would otherwise stop where the span was planned and write that stale
        snapshot as the current state: the bank once said the Telegram bot was not set up,
        written eight minutes after it was."""
        if planned_end.time() == datetime.min.time():
            return planned_end
        midnight_after = datetime.combine(planned_end.date() + timedelta(days=1), datetime.min.time())
        return max(planned_end, min(midnight_after, now or datetime.now()))

    @classmethod
    def _curate_period(cls, curator: 'LongTermMemoryCurator', period_start, period_end, history,
                       record: bool = True) -> dict:
        """One period, lock already held: curate, run the commands, save the bank and its debug
        log and, when `record`, leave the completion in the history. The commands run before
        the bank is written: the bank keeps only a pointer to a how-to the commands move into
        a procedure, so a refused `save_procedure` must be known before the bank is replaced —
        its text is then kept in the bank. A period with no interactions changes nothing."""
        new_memory, metadata = curator.curate_memory(period_start, period_end)
        if metadata.get('empty'):
            info("Rem: nothing happened in this period — memory bank unchanged.")
            return metadata
        metadata['command_results'], unsaved = curator.run_commands(metadata.get('commands', ''))
        if unsaved:
            new_memory = "\n\n".join([new_memory.rstrip("\n")] + unsaved) + "\n"
            metadata['new_memory_size'] = len(new_memory)
        # Redacted once here: the bank's write redacts on its own, the debug copy must not
        # keep what the bank was not allowed to.
        new_memory = redact_for_storage(new_memory)
        curator.save_memory(new_memory, metadata)
        curator.save_debug_log(new_memory, metadata)
        info(f"Rem: cycle complete — dreamt through {metadata['interaction_count']} interactions, "
             f"memory bank now {metadata['new_memory_size']} bytes.")
        if history is None:
            trace("Rem: no history to record the completion in")
        elif not record:
            trace("Rem: one chunk of a span — the span's row records the completion")
        else:
            history.add_context_message("system", cls.completion_note(metadata), time.time(),
                                        source=HISTORY_SOURCE)
        return metadata

    @staticmethod
    def span_note(summary: dict) -> str:
        """The system row a finished multi-chunk span leaves in the dialog history: the span and
        what it learnt from; the chunks that failed are mentioned only when there were any."""
        start, end = (datetime.fromisoformat(summary[k]) for k in ('period_start', 'period_end'))
        text = (f"Long term memory curated over {start.day} {start:%B} .. {end.day} {end:%B} "
                f"from {summary['interaction_count']} interactions")
        if summary['failed']:
            text += f"; {summary['failed']} of its {summary['chunks']} chunks failed and left nothing"
        return text + "."

    @staticmethod
    def completion_note(metadata: dict) -> str:
        """The system row a finished curation leaves in the dialog history."""
        if metadata.get('truncated'):
            note = "Long term memory curation aborted - the AI's reply was cut short, memory unchanged"
        else:
            note = "Long term memory curation complete - memory updated"
        note += f" ({metadata.get('interaction_count', 0)} interactions"
        if metadata.get('period_start'):
            note += f", {metadata['period_start']} .. {metadata['period_end']}"
        return note + ")."

    def curate_memory(self,
                      period_start: datetime | date = None,
                      period_end: datetime | date = None,
                      lookback_days: int = 2,
                      lookahead_days: int = 5):
        """
        Run memory curation over the interactions of a period.

        Args:
            period_start: First moment of the period (default: midnight `lookback_days` before period_end)
            period_end: Last moment of the period, exclusive (default: now)
            lookback_days: How many days back the default period_start reaches
            lookahead_days: How many days of calendar to include (default: 5)

        A bare date means midnight at its start. Both ends are accepted as dates or datetimes so a
        nightly run says "the last two days" and a first-run curation says "since 14:03".

        Returns:
            Tuple of (updated_memory_text, metadata_dict)
            metadata_dict contains: {
                'period_start': str, 'period_end': str,
                'interaction_count': int,
                'calendar_events_count': int,
                'old_memory_size': int,
                'new_memory_size': int,
                'llm_duration_seconds': float,
                'timestamp': float
            }
        """
        period_end = self._as_datetime(period_end) if period_end else datetime.now()
        if period_start:
            period_start = self._as_datetime(period_start)
        else:
            period_start = datetime.combine(period_end.date() - timedelta(days=lookback_days), datetime.min.time())
        if period_start >= period_end:
            raise ValueError(f"Memory curation period is empty: {period_start} .. {period_end}")

        info(f"Rem: entering REM sleep — dreaming over {period_start:%Y-%m-%d %H:%M} .. {period_end:%Y-%m-%d %H:%M}...")

        from ..commands.Commands import is_reference_result      # Commands hosts the curator's own commands
        interaction_history = self.floxy_db.get_interaction_history(period_start, period_end,
                                                                    skip_command_result=is_reference_result)
        trace(f"Loaded {len(interaction_history)} bytes of interaction history.")
        if interaction_history == NO_INTERACTIONS:
            info(f"Rem: no interactions between {period_start:%Y-%m-%d %H:%M} and {period_end:%Y-%m-%d %H:%M} "
                 "— nothing to dream about, the bank stays as it is.")
            current_memory = self._load_memory()
            return current_memory, {
                'period_start': period_start.isoformat(timespec='minutes'),
                'period_end': period_end.isoformat(timespec='minutes'),
                'interaction_count': 0, 'calendar_events_count': 0,
                'old_memory_size': len(current_memory), 'new_memory_size': len(current_memory),
                'llm_duration_seconds': 0.0, 'truncated': False, 'empty': True, 'timestamp': time.time(),
            }

        calendar_start = period_end.date()
        calendar_end = calendar_start + timedelta(days=lookahead_days)
        calendar_data = CalendarAccess.current().fetch(calendar_start, calendar_end)
        trace(f"Loaded {len(calendar_data)} bytes of calendar data.")

        current_memory = self._load_memory()
        trace(f"Loaded {len(current_memory)} bytes of long term memory.")

        return self._curate_with_data(interaction_history, calendar_data, current_memory,
                                      period=(period_start, period_end))

    @staticmethod
    def word_count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def memory_size_note(word_count: int) -> str:
        """The measured size of the bank, stated so the LLM does not have to estimate it."""
        note = f"The current memory bank is {word_count} words (the limit is {MEMORY_WORD_LIMIT} words)."
        if word_count > MEMORY_WORD_LIMIT:
            excess = word_count - MEMORY_WORD_LIMIT
            note += (f" It is OVER the limit by {excess} words: the updated bank you output must be at least "
                     f"{excess} words shorter than the current one, on top of any additions you make.")
            if excess > MEMORY_WORD_LIMIT * 0.1:
                note += (" Condensing the bank is a REQUIRED part of this run, not optional: you MUST condense "
                         "to meet the limit. Removing the least important information is acceptable.")
        else:
            note += f" You have {MEMORY_WORD_LIMIT - word_count} words of headroom for additions."
        return note

    @staticmethod
    def _as_datetime(value: datetime | date) -> datetime:
        if isinstance(value, datetime):
            return value
        return datetime.combine(value, datetime.min.time())

    def curate_memory_with_data(self,
                                 interaction_history: str,
                                 calendar_data: str,
                                 current_memory: str):
        """
        Run memory curation with provided data (for testing).

        Args:
            interaction_history: Recent interactions text
            calendar_data: Calendar events text
            current_memory: Current memory bank text

        Returns:
            Tuple of (updated_memory_text, metadata_dict)
        """
        return self._curate_with_data(interaction_history, calendar_data, current_memory)

    def _curate_with_data(self,
                          interaction_history: str,
                          calendar_data: str,
                          current_memory: str,
                          period: tuple[datetime, datetime] | None = None):
        """
        Internal method to run curation with provided data.

        Args:
            interaction_history: Recent interactions text
            calendar_data: Calendar events text
            current_memory: Current memory bank text
            period: (start, end) the history was taken from; None describes it as "the recent interactions"

        Returns:
            Tuple of (updated_memory_text, metadata_dict)
        """
        # Count interactions and calendar events
        interaction_count = len([l for l in interaction_history.splitlines() if l.startswith("[")])
        calendar_events_count = 0 if calendar_data in (NO_EVENTS, EVENTS_UNAVAILABLE, NOT_CONFIGURED) else len(calendar_data.splitlines())

        if period:
            period_text = f"{period[0]:%Y-%m-%d %H:%M} to {period[1]:%Y-%m-%d %H:%M}"
        else:
            period_text = "the recent interactions below"

        voice_enabled = bool(ConfigManager.get_instance().get('interfaces.voice.enabled', False))
        trace(f"Memory curation prompt: STT note {'included' if voice_enabled else 'skipped'} "
              f"(interfaces.voice.enabled={voice_enabled})")
        word_count = self.word_count(current_memory)
        info(f"Rem: current memory bank is {word_count} words (limit {MEMORY_WORD_LIMIT})")
        full_prompt = PROMPT.replace("{WATCHLIST}", WatchList.render(self.watchlist_dir)) \
            .replace("{PROCEDURES}", Procedures.render_index(self.procedures_dir)) \
            .replace("{TODAY}", f"{datetime.now():%Y-%m-%d}") \
            .replace("{INTERACTION_HISTORY}", interaction_history) \
            .replace("{CALENDAR}", calendar_data) \
            .replace("{CURRENT_MEMORY}", current_memory) \
            .replace("{MEMORY_SIZE}", self.memory_size_note(word_count)) \
            .replace("{WORD_LIMIT}", str(MEMORY_WORD_LIMIT)) \
            .replace("{PERIOD}", period_text) \
            .replace("{STT_NOTE}", STT_NOTE if voice_enabled else "") \
            .replace("{TIPS}", self._tips_prompt())

        # Send to LLM with thinking enabled for complex reasoning
        info("Rem: deep in REM — consolidating the period's memories with the AI...")
        start_time = time.time()
        response = self._think(full_prompt, "Please curate the memory now.")
        llm_duration = time.time() - start_time
        trace(f"Rem: woke after {llm_duration:.1f}s of dreaming")

        response, commands = self._split_commands(response)
        new_memory = self._extract_memory_content(response)
        new_words = self.word_count(new_memory)
        info(f"Rem: the AI returned a {new_words}-word memory bank (was {word_count}, limit {MEMORY_WORD_LIMIT})"
             + (" — still over the limit" if new_words > MEMORY_WORD_LIMIT else ""))

        # A reply cut off at the output cap is a bank missing its tail: saving it would
        # forget every section past the cut, and the next run could never bring them back.
        truncated = getattr(self.llm, 'last_stop_reason', None) == 'max_tokens'
        if truncated:
            error(f"Rem: the dream was cut short — the AI hit its output limit after {len(new_memory)} chars. "
                  "Keeping the previous memory bank.")
            new_memory = current_memory
            commands = ""
        elif not new_memory or len(new_memory) < len(current_memory) * 0.3:
            error(f"WARNING: AI returned suspiciously short memory ({len(new_memory)} chars)")
            error("Keeping previous memory state")
            new_memory = current_memory
            commands = ""

        # Build metadata
        metadata = {
            'period_start': period[0].isoformat(timespec='minutes') if period else '',
            'period_end': period[1].isoformat(timespec='minutes') if period else '',
            'interaction_count': interaction_count,
            'calendar_events_count': calendar_events_count,
            'old_memory_size': len(current_memory),
            'new_memory_size': len(new_memory),
            'llm_duration_seconds': llm_duration,
            'truncated': truncated,
            'timestamp': time.time(),
            'old_memory': current_memory,
            'full_prompt': full_prompt,
            'commands': commands,
        }

        return new_memory, metadata

    def _think(self, prompt: str, user_message: str) -> str:
        """The curation call: through the batch API at half price when this run can wait
        (`batch`) and `ai.memory_curator.batch` has not turned it off, the ordinary call
        otherwise — and whenever the batch gave no answer (a backend without a batch API, the
        wait ran out, an error). A refusal is not retried: the same request is refused again."""
        config = ConfigManager.get_instance()
        if not self.batch:
            trace("Rem: an interactive curation — the ordinary call")
        elif not config.get('ai.memory_curator.batch', True):
            info("Rem: ai.memory_curator.batch is off — the ordinary call")
        else:
            wait_minutes = float(config.get('ai.memory_curator.batch_wait_minutes', BATCH_WAIT_MINUTES))
            try:
                response = self.llm.think_batched(system_message=prompt, user_message=user_message,
                                                  thinking=True, max_wait_seconds=wait_minutes * 60)
            except LLMRefusalError:
                raise
            except Exception as e:
                error(f"Rem: the batch failed ({type(e).__name__}: {e})")
                response = None
            if response is not None:
                info("Rem: the dream came back from the batch API (half price)")
                return response
            info("Rem: no answer from the batch API — making the ordinary call")
        return self.llm.think(system_message=prompt, user_message=user_message, thinking=True)

    def save_memory(self, memory_text: str, metadata: dict = None) -> str:
        """
        Save memory to disk.

        The bank was loaded minutes ago (metadata['old_memory']); anything appended to the file
        since — a `remember` while the curator was thinking — is carried over onto the new bank
        rather than lost. The read-merge-write happens under the bank's lock so no append
        can slip in between.

        Args:
            memory_text: The memory content to save
            metadata: Optional metadata dict for logging

        Returns:
            Path to saved file
        """
        try:
            with MemoryFile.lock:
                loaded = (metadata or {}).get('old_memory')
                if loaded is not None:
                    on_disk = self._load_memory()
                    if on_disk != loaded:
                        if on_disk.startswith(loaded):
                            appended = on_disk[len(loaded):]
                            info(f"Rem: memory grew by {len(appended)} chars mid-dream — carrying the additions over.")
                            memory_text = memory_text.rstrip("\n") + "\n" + appended
                        else:
                            error("Memory file was rewritten during curation — the curated bank replaces it.")
                MemoryFile.write(memory_text, self.memory_file_path)
            info(f"Rem: memory consolidated to {self.memory_file_path} ({len(memory_text)} chars)")
            return self.memory_file_path
        except Exception as e:
            error(f"Error saving memory to {self.memory_file_path}: {e}")
            raise

    @staticmethod
    def save_debug_log(memory_text: str, metadata: dict) -> str:
        """
        Save memory with metadata to debug log.

        Saves to: AIServer/log/memory/<timestamp>_<guid>_memory.md

        Returns:
            Path to debug log file
        """
        try:
            # Create log directory if it doesn't exist
            log_dir = Path("log/memory")
            log_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename with timestamp and guid
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            guid = uuid.uuid4().hex[:8]
            filename = f"{timestamp}_{guid}_memory.md"
            log_path = log_dir / filename

            # Generate diff if old memory exists
            diff_text = ""
            old_memory = metadata.get('old_memory', '')
            if old_memory:
                old_lines = old_memory.splitlines(keepends=True)
                new_lines = memory_text.splitlines(keepends=True)
                diff = difflib.unified_diff(
                    old_lines,
                    new_lines,
                    fromfile='previous_memory.md',
                    tofile='updated_memory.md',
                    lineterm=''
                )
                diff_text = ''.join(diff)

            # Write memory with metadata header
            with open(log_path, 'w') as f:
                f.write("# Memory Debug Log\n\n")
                f.write(f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if metadata.get('period_start'):
                    f.write(f"- Period: {metadata['period_start']} .. {metadata['period_end']}\n")
                f.write(f"- Interactions: {metadata.get('interaction_count', 0)}\n")
                f.write(f"- Calendar events: {metadata.get('calendar_events_count', 0)}\n")
                f.write(f"- Old size: {metadata.get('old_memory_size', 0)} chars\n")
                f.write(f"- New size: {metadata.get('new_memory_size', 0)} chars\n")
                f.write(f"- LLM duration: {metadata.get('llm_duration_seconds', 0):.1f}s\n")
                if metadata.get('truncated'):
                    f.write("- Result: reply cut off at the output cap — previous bank kept\n")
                f.write("\n")
                f.write("---\n\n")

                full_prompt = metadata.get('full_prompt', '')
                if full_prompt:
                    f.write("## LLM Prompt\n\n")
                    f.write("```\n")
                    f.write(full_prompt)
                    f.write("\n```\n\n")
                    f.write("---\n\n")

                if metadata.get('command_results'):
                    f.write("## Watch list and procedure commands\n\n")
                    f.write("\n".join(f"- {line}" for line in metadata['command_results']))
                    f.write("\n\n---\n\n")

                if diff_text:
                    f.write("## Diff from Previous Memory\n\n")
                    f.write("```diff\n")
                    f.write(diff_text)
                    f.write("\n```\n\n")
                    f.write("---\n\n")

                f.write("## Updated Memory\n\n")
                f.write(memory_text)

            info(f"Rem: dream journal saved to {log_path}")
            return str(log_path)

        except Exception as e:
            error(f"Error saving debug log: {e}")
            return ""

    def _load_memory(self) -> str:
        return MemoryFile.read(self.memory_file_path)

    @staticmethod
    def _split_commands(response: str) -> tuple[str, str]:
        """The reply without its <commands> block, and the block's content ('' when there is
        none). An unclosed block runs to the end of the reply."""
        match = re.search(r'<commands>(.*?)(?:</commands>|\Z)', response, re.DOTALL)
        if not match:
            return response, ""
        return response[:match.start()] + response[match.end():], match.group(1).strip()

    def _think_whole(self, prompt: str, what: str) -> str:
        """One LLM call whose reply must be complete: a reply cut off at the output cap raises."""
        response = self.llm.think(system_message=prompt, user_message="Please do it now.", thinking=True)
        if getattr(self.llm, 'last_stop_reason', None) == 'max_tokens':
            raise RuntimeError(f"the AI's reply was cut off at its output limit while {what}")
        return response

    def propose_split(self, target: Path) -> dict:
        """Split the current bank into the three parts, as a proposal under `target`, laid out
        like `memory/` (bank.md, watchlist/, procedures/) so it can be read and edited by hand
        before `apply_split` puts it in place. Nothing live is touched. Returns the counts."""
        bank = self._load_memory()
        if not bank.strip():
            raise RuntimeError(f"there is no memory bank at {self.memory_file_path} to split")
        today = f"{datetime.now():%Y-%m-%d}"
        watchlist_dir, procedures_dir = target / "watchlist", target / "procedures"
        info(f"Memory split: lifting the watch items and procedures out of a {self.word_count(bank)}-word bank...")
        reply = self._think_whole(SPLIT_EXTRACT_PROMPT.replace("{TODAY}", today).replace("{BANK}", bank),
                                  "extracting the watch items and procedures")
        commands = self._split_commands(reply)[1]
        # A refused procedure stays in the old bank's text, which the rewrite below keeps verbatim
        results, _ = self.run_commands(commands, watchlist_dir, procedures_dir, log_dir=target / "replaced")
        info(f"Memory split: {len(WatchList.items(watchlist_dir))} watch item(s), "
             f"{len(Procedures.names(procedures_dir))} procedure(s); rewriting the bank without them...")
        procedures = "\n\n".join(Procedures.load(name, procedures_dir) for name in Procedures.names(procedures_dir))
        reply = self._think_whole(SPLIT_BANK_PROMPT.replace("{TODAY}", today)
                                  .replace("{WATCHLIST}", WatchList.render(watchlist_dir))
                                  .replace("{PROCEDURES}", procedures or Procedures.EMPTY)
                                  .replace("{BANK}", bank),
                                  "rewriting the bank")
        new_bank = self._extract_memory_content(reply)
        if not new_bank:
            raise RuntimeError("the AI returned an empty bank")
        target.mkdir(parents=True, exist_ok=True)
        (target / "bank.md").write_text(new_bank.rstrip("\n") + "\n", encoding="utf-8")
        (target / "commands.log").write_text("\n".join(results) + "\n", encoding="utf-8")
        summary = {'target': str(target), 'old_words': self.word_count(bank), 'new_words': self.word_count(new_bank),
                   'watch_items': len(WatchList.items(watchlist_dir)),
                   'procedures': len(Procedures.names(procedures_dir)),
                   'failed_commands': sum(r.startswith(('FAILED', 'SKIPPED')) for r in results)}
        info(f"Memory split: proposal in {target} — bank {summary['old_words']} → {summary['new_words']} words, "
             f"{summary['watch_items']} watch item(s), {summary['procedures']} procedure(s), "
             f"{summary['failed_commands']} command(s) failed or skipped")
        return summary

    def apply_split(self, source: Path) -> dict:
        """Put a proposal in place: the live memory is copied to log/memory/ first, then the
        bank is replaced and the proposal's watch items and procedures are copied in; an item
        or a procedure that already exists is left as it is."""
        bank_file = source / "bank.md"
        if not bank_file.is_file():
            raise RuntimeError(f"{source} holds no bank.md — not a split proposal")
        backup = MEMORY_LOG_DIR / f"before_split_{datetime.now():%Y%m%d_%H%M%S}"
        if MEMORY_DIR.is_dir():
            shutil.copytree(MEMORY_DIR, backup)
            info(f"Memory split: the live memory is backed up in {backup}")
        else:
            info(f"Memory split: no {MEMORY_DIR} to back up")
        MemoryFile.write(bank_file.read_text(encoding="utf-8"), self.memory_file_path)
        copied, kept = {}, {}
        stores = (("watchlist", self.watchlist_dir or WatchList.WATCHLIST_DIR),
                  ("procedures", self.procedures_dir or Procedures.PROCEDURES_DIR))
        for store, directory in stores:
            copied[store], kept[store] = [], []
            for file in sorted((source / store).glob("*.md")):
                if (directory / file.name).exists():
                    kept[store].append(file.stem)
                    continue
                directory.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(file, directory / file.name)
                copied[store].append(file.stem)
        already = [name for names in kept.values() for name in names]
        info(f"Memory split applied from {source}: bank replaced, {len(copied['watchlist'])} watch item(s) and "
             f"{len(copied['procedures'])} procedure(s) copied"
             + (f", {len(already)} already existed and were kept: {', '.join(already)}" if already else ""))
        return {'source': str(source), 'backup': str(backup), 'watch_items': copied['watchlist'],
                'procedures': copied['procedures'], 'kept': already}

    _COMMAND_START = re.compile(r'^[`\-*•\s]*(' + '|'.join(CURATOR_COMMANDS + (TIPS_COMMAND,)) + r')\s*\(',
                                re.IGNORECASE)

    @staticmethod
    def tips_destination() -> str:
        """The group the run posts tips to (`ai.memory_curator.tips_destination`), '' when off."""
        return str(ConfigManager.get_instance().get(TIPS_KEY, '') or '').strip()

    def _tips_prompt(self) -> str:
        destination = self.tips_destination()
        if not destination:
            trace("Rem: no tips destination configured, the prompt says nothing about tips")
            return ""
        return TIPS_PROMPT.replace("{TIPS_DESTINATION}", destination) \
            .replace("{OWNER}", ConfigManager.get_instance().get_owner_name())

    def _post_tip(self, params: list[str]) -> str:
        """Deliver a `post_message` of the command block to the configured tips group. Returns ''
        or the reason it was refused. The text is vault-redacted first — the bank's write
        redacts on its own, a post leaves the house."""
        destination = self.tips_destination()
        if not destination:
            return "tips are off (ai.memory_curator.tips_destination is not set)"
        if len(params) < 2:
            return "post_message failed: expected 'destination, text'"
        # An under-quoted text with commas arrives in pieces: put it back together.
        text = ", ".join(part.strip() for part in params[1:] if part.strip())
        if not text:
            return "post_message failed: the text is empty"
        asked = params[0].strip().strip("\"'`")
        if asked.lower() != destination.lower():
            info(f"Rem: the tip named {asked!r} as its destination — posted to the configured {destination!r} instead")
        failure = Destinations.send(destination, redact_for_storage(text))
        if not failure:
            info(f"Rem: tip posted to {destination!r} ({len(text)} chars)")
        return failure

    @classmethod
    def _command_chunks(cls, block: str) -> tuple[list[str], list[tuple[str, str]]]:
        """The block cut at every line that opens a curator command (outside a triple-quoted
        text): the lines that belong to no command, and per command its name and the raw text
        between its opening parenthesis and the last closing one. A command runs on to the next
        line only while it is open — inside a triple-quoted text, or with its parentheses not
        yet balanced (quotes do not count: an inch mark or a quoted phrase in a note must not
        swallow the commands after it); any other line is skipped, so a forbidden command or a
        line of prose never ends up inside the previous command's last argument."""
        skipped, chunks, current, in_text, depth = [], [], None, False, 0
        for line in block.splitlines():
            start = None if in_text else cls._COMMAND_START.match(line)
            if start:
                current = [start.group(1).lower(), line[start.end():]]
                chunks.append(current)
                in_text, depth = cls._scan(current[1], in_text, 1)
            elif current is not None and (in_text or depth > 0):
                current[1] += "\n" + line
                in_text, depth = cls._scan(line, in_text, depth)
            else:
                in_text, depth = cls._scan(line, in_text, 0)
                if line.strip("` "):
                    skipped.append(line.strip())
            if current is not None and not in_text and depth <= 0:
                current = None
        return skipped, [(name, raw[:raw.rfind(")")] if ")" in raw else raw) for name, raw in chunks]

    @staticmethod
    def _scan(text: str, in_text: bool, depth: int) -> tuple[bool, int]:
        """`in_text` and the parenthesis depth after `text`; parentheses count only outside a
        triple-quoted text."""
        for index, part in enumerate(text.split('"""')):
            if index:
                in_text = not in_text
            if not in_text:
                depth += part.count("(") - part.count(")")
        return in_text, depth

    @staticmethod
    def _split_arguments(raw: str) -> list[str]:
        """The arguments of one curator command. A triple-quoted argument runs to its closing
        triple quote; a quoted one ends only at a quote that is followed by a comma or the end,
        so a quote inside the text does not cut it short; anything else runs to the next comma."""
        args, i, n = [], 0, len(raw)
        while i < n:
            while i < n and raw[i] in " \t\r\n":
                i += 1
            if i >= n:
                break
            if raw.startswith('"""', i):
                end = raw.find('"""', i + 3)
                end = n if end < 0 else end
                args.append(raw[i + 3:end].strip("\n"))
                i = end + 3
            elif raw[i] in "\"'`":
                quote, j = raw[i], i + 1
                while True:
                    j = raw.find(quote, j)
                    if j < 0 or not raw[j + 1:].strip() or raw[j + 1:].lstrip().startswith(","):
                        break
                    j += 1
                end = n if j < 0 else j
                args.append(raw[i + 1:end].replace("\\" + quote, quote))
                i = end + 1
            else:
                end = raw.find(",", i)
                end = n if end < 0 else end
                args.append(raw[i:end].strip())
                i = end
            while i < n and raw[i] != ",":
                i += 1
            i += 1
        return args

    def run_commands(self, block: str, watchlist_dir: Path = None, procedures_dir: Path = None,
                     log_dir: Path = None) -> tuple[list[str], list[str]]:
        """Run the curator's command block against the watch list and the procedures, one
        command after the other; a line that is not one of CURATOR_COMMANDS is skipped. A
        `save_watch` here is a rewrite, not a look, so it leaves the item's `checked` stamp alone.
        Returns one line per command for the debug log (the command, and why it failed if it
        did), and the bank sections that keep the text of every refused `save_procedure`: the
        bank is rewritten to hold a one-line pointer in its place, so the how-to would be lost
        otherwise. The directories default to the curator's own; `propose_split` points them at
        its proposal."""
        watchlist_dir = watchlist_dir or self.watchlist_dir
        procedures_dir = procedures_dir or self.procedures_dir
        if not block.strip():
            trace("Rem: no watch list or procedure commands this run")
            return [], []
        results, unsaved = [], []
        skipped, chunks = self._command_chunks(block)
        for line in skipped:
            info(f"Rem: skipped a line that is not a curator command: {line[:120]}")
            results.append(f"SKIPPED (not a curator command): {line[:200]}")
        for command, raw in chunks:
            params = self._split_arguments(raw)
            shown = f"{command}({', '.join(p if len(p) <= 120 else p[:120] + '…' for p in params)})"
            if command == TIPS_COMMAND and not self.tips_destination():
                info(f"Rem: skipped a post_message — tips are off: {shown[:120]}")
                results.append(f"SKIPPED (tips are off): {shown[:200]}")
                continue
            if command == TIPS_COMMAND:
                failure = self._post_tip(params)
            elif command in WatchList.COMMANDS:
                failure = WatchList.run(command, params, watchlist_dir, log_dir, looked=False)
            else:
                failure = Procedures.run(command, params, procedures_dir, log_dir, allow_replace=False)
            if failure:
                error(f"Rem: {shown} — {failure}")
                results.append(f"FAILED {shown} — {failure}")
                if command == "save_procedure" and len(params) > 2 and "".join(params[2:]).strip():
                    unsaved.append(f"## Procedure not saved: {params[0]}\n"
                                   f"({failure}; move it with the commands)\nWhen: {params[1]}\n\n"
                                   + ", ".join(params[2:]).strip())
                    info(f"Rem: the text of '{params[0]}' stays in the bank until it is saved")
            else:
                info(f"Rem: {shown}")
                results.append(shown)
        info(f"Rem: {len(results)} watch list / procedure command(s) handled, "
             f"{sum(r.startswith('FAILED') for r in results)} failed")
        return results, unsaved

    @staticmethod
    def _extract_memory_content(response: str) -> str:
        """Extract memory content from AI response, handling markdown code blocks."""
        tagged = re.search(r'<memory>(.*?)(?:</memory>|\Z)', response, re.DOTALL)
        if tagged:
            return tagged.group(1).strip()
        # Try to extract from markdown code block
        match = re.search(r'```(?:markdown)?\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Otherwise use the response as-is (strip leading/trailing whitespace)
        return response.strip()
