import re
import shutil
import time
import uuid
import difflib
import threading
from datetime import datetime, timedelta, date
from pathlib import Path

from .. import Destinations, MemoryFile, WatchList
from ..Memos import Memos
from ..Procedures import Procedures
from ..llm import create_llm, parse_llm_config
from .MemoryCondenser import MemoryCondenser
from common.ConfigManager import ConfigManager
from common.FloxyDatabase import FloxyDatabase, NO_INTERACTIONS
from assistant.integration.calendar.CalendarAccess import CalendarAccess, EVENTS_UNAVAILABLE, NOT_CONFIGURED
from assistant.integration.calendar.Event import NO_EVENTS
from common.logging import info, error, trace
from ..llm.AbstractLLM import LLMRefusalError, Thinking
from ..llm.LLMUsage import LLMUsage
from common.Paths import MEMORY_DIR, MEMORY_LOG_DIR
from common.vault.Redactor import redact_for_storage

HISTORY_SOURCE = "memory_curation"   # the `source` of the system row a finished curation leaves


# A span longer than this is curated in chunks of this many days, one after another, each
# reading the bank the previous one wrote (`run_span`).
CURATION_CHUNK_DAYS = 4
# The size the bank is kept around. The code keeps it (the condensing pass below); the curation
# prompt names no size at all: a model given a count and a limit tries to meet it by counting
# words — 64000 output tokens of thinking twice in one night (2026-11-13), no bank written.
MEMORY_WORD_LIMIT = 4000
# A curated bank more than CONDENSE_TRIGGER over the limit is condensed after the curation
# (`MemoryCondenser`, section by section) and asked down to CONDENSE_AIM under the limit (`condense_cut`) —
# under it, because the model gives a part of what it is asked for and never all of it
# (2026-11-17: a 5283-word bank asked for 1783 lost 744). While the bank is still over the
# trigger the pass runs again, CONDENSE_PASSES times at most: each pass gets about half of the
# one before (468, 196, 103 that day), so a fourth buys little. A pass that removes less than
# CONDENSE_MIN_DELIVERY of its ask ends the loop with an error — the next would do no better.
# The curation prompt cannot do the cutting: with the
# period's interactions and the "never drop a fact" rules in front of it the model rewords an
# oversize bank at the same length however plainly it is told to cut — three nights
# at ~5000 words, and a test pass with the full prompt cut 8 words of 992 asked. After, not
# before: a pass before the curation lands the bank on the limit and the curation then adds the
# period's facts on top of it (a 5269-word bank condensed to 4020 came back at 4357), so the bank
# saved was over the limit every night; condensed last, what is saved is the condensed bank.
# Trigger above the limit, aim below it, both as a share of it: a hysteresis, so a condensed
# bank takes a few nights to grow back to the trigger instead of being condensed every night.
CONDENSE_TRIGGER = 0.10
CONDENSE_AIM = 0.125
CONDENSE_PASSES = 3
CONDENSE_MIN_DELIVERY = 0.10
# How long a nightly run waits for its batch before it makes the ordinary full-price call
# (`ai.memory_curator.batch_wait_minutes`): from the 04:00 run, done before the house wakes.
BATCH_WAIT_MINUTES = 90
# What the curator may do to the watch list, the procedures and the memos, in the command block
# after the bank. It may add to a procedure but never replace or delete one: a procedure is
# rewritten only by the assistant that has the working flow in front of it, or by the user.
# A memo it keeps current, so it may replace and delete one.
# Not `checked_watch`: only a real check may stamp an item as looked at.
CURATOR_COMMANDS = ("save_watch", "delete_watch", "save_procedure", "procedure_note", "save_memo", "delete_memo")
# The commands that move text out of the bank: one that is refused keeps its text there.
MOVES = {"save_procedure": "Procedure", "save_memo": "Memo"}
# Under this a block is not worth a memo: every memo costs its index line in every prompt.
MEMO_MIN_WORDS = 100
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
update the long-term memory. This run covers {PERIOD}. Today is {TODAY}.{FOLLOWS}

The long-term memory has four parts:
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
- the **memos**: what there is to know about one subject, for when that subject comes up — a trip's bookings, a list
  of wines, test results, a directory of tradespeople. The AI sees their index and loads one when it needs it. You see
  them in full and keep them current.

{STT_NOTE}

## Recent interactions:
{INTERACTION_HISTORY}

## Current memory bank:
{CURRENT_MEMORY}

## Current watch list (open items):
{WATCHLIST}

## Procedures (index only):
{PROCEDURES}

## Memos (in full):
{MEMOS}

{IN_PROMPT}

## Your task:
Update the memory bank based on new information. You may add, modify, remove and restructure information as needed.
Keep the memory bank concise: a fact the period supersedes is replaced, not kept beside the new one, and what is outdated goes. Its size is not yours to manage - do not count words.

Then bring the watch list and the memos in line with what the period shows, with commands (see the output format).

Rules:
- The bank does NOT include command names or action IDs (the AI knows what commands it can run)
- The bank does NOT include command syntax or how to interpret commands (the AI already understands natural language and has extensive knowledge of available commands). A procedure is different: there the exact commands and code that worked are the point
- Do NOT include song/playlist/favorite IDs
- Focus on personal preferences, patterns, and facts about the family that help personalization
- The bank is about the family and their world, never about the Keep software. Leave out how the system works, what it
  can and cannot do, its bugs and workarounds, etc. All of that changes with every release, and the AI has its own
  documentation for it
- When a user corrects a misheard command (e.g. by saying "No, I said X" right after), the original action was based on a MISHEARD phrase. Record the STT correction, but do NOT add anything from the misinterpreted action as a genuine preference
- Be very mindful of making assumptions - errors sneak easily into the memory bank and can propagate over time - so only enter things you are sure of in the memory
- Remove a fact only when it of extremely low value or recent interactions give positive evidence against it: contradicted, superseded, explicitly retracted, or expired. Not being mentioned in this period's logs is not necessarily staleness
- The bank and the memos are the system's ONLY long-term storage of facts — anything dropped from both is permanently forgotten. Be mindful of deleting valuable information
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
- The bank holds no lookup material: facts needed only when their subject comes up belong in a memo. Move such a
  block of about {MEMO_MIN_WORDS} words or more with `save_memo(...)`, every detail kept, and leave at most a one-line
  pointer in the bank. What must act unasked stays in the bank even when the facts around it move: who is who, a rule,
  a sensitivity, a secrecy clause, a preference that shapes an answer nobody asked about
- Memo upkeep: write what the period adds or changes into the memo it belongs to (`save_memo` under its existing name
  — the text REPLACES the old one, so carry over every detail that still holds); delete a memo whose subject is over
  (`delete_memo`), first moving what is durably true of it to the bank. Leave a memo alone when the period says
  nothing about it
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
  "weekly"), optional — omitted, both are kept as they were. A note longer than {WATCH_WORD_LIMIT} words is refused
  and the item keeps its old note. Never `checked_watch`: only a real check stamps it
- `delete_watch("name", "outcome")` — the item is over
- `save_procedure("name", "when to load it", \"\"\"the full text\"\"\")` — a NEW procedure; the text in a
  triple-quoted block. Keep `when` under  {PROCEDURE_WHEN_CHARS} characters
- `procedure_note("name", "the lesson")` — append a lesson to an existing procedure
- `save_memo("name", "when to load it", \"\"\"the full text\"\"\")` — a new memo, or the whole new text of the memo
  with that name; `when` under {PROCEDURE_WHEN_CHARS} characters, the text under {MEMO_MAX_CHARS}
- `delete_memo("name")` — the memo's subject is over
{TIPS}"""

# Appended to the prompt only while `ai.memory_curator.tips_destination` is set.
# Said to every chunk of a span but the last: what it reads as open may be settled in the days it does not see.
FOLLOWS_NOTE = " The days between this period and today are curated in the runs that follow this one."

TIPS_PROMPT = """
## Tips for other Keeps
Other households run their own Keep, and the Keeps share tips in the Telegram group "{TIPS_DESTINATION}". 
Tips are things that other Keep households will find valuable that took you and your user time to figure out. 
For example procedures that can be shared:
 * How to use Klarna login (popup only works when the owner's phone is on the home network, etc)
 * How to deliver BankID autostart links
 * How to use Shopify checkouts (card fields in the pci iframes are fillable, etc)
Tips are plain-language, in the command block:
`post_message("{TIPS_DESTINATION}", \"\"\"Tip from {OWNER}'s Keep: ...\"\"\")` — the prefix names whose Keep it comes from.
Be mindful of privacy: the group is read by other households. The tip is posted to a Telegram chat so keep it short and use nice telegram formatting. As short at possible, NEVER over 2000 chars.
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
- `save_procedure("name", "when to load it", \"\"\"the full text\"\"\")` — one per how-to, the text in a
  triple-quoted block; keep `when` under {PROCEDURE_WHEN_CHARS} characters. Copy every step, URL, endpoint, code snippet, number and warning VERBATIM from the bank; do not
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
    once the system is settled, more often while it is new — and tidies the watch list, adds to
    the procedures and keeps the memos through the command block that follows the bank in its
    reply. The memos are where the bank gets shorter: lookup material moves out of it into them."""

    _curation_lock = threading.Lock()
    _backfilling = False

    @classmethod
    def backfilling(cls) -> bool:
        """True while a span of several chunks is between its first chunk and its last: the
        watch list is then that of a past day — items the chunks still to come will settle —
        and nothing should act on it."""
        return cls._backfilling

    def __init__(self,
                 floxy_db: FloxyDatabase = None,
                 memory_file_path: str = None,
                 llm=None,
                 test_mode: bool = False,
                 watchlist_dir: Path = None,
                 procedures_dir: Path = None,
                 batch: bool = False,
                 memos_dir: Path = None):
        """
        Initialize the memory curator.

        Args:
            floxy_db: FloxyDatabase instance (defaults to FloxyDatabase.get_instance())
            memory_file_path: Path to the memory bank (defaults to memory/bank.md)
            llm: LLM instance to use (defaults to creating ClaudeLLM)
            test_mode: If True, skip database initialization (use curate_memory_with_data only)
            watchlist_dir: The watch list the curator reads and tidies (defaults to memory/watchlist)
            procedures_dir: The procedures it indexes and adds to (defaults to memory/procedures)
            memos_dir: The memos it reads in full and keeps current (defaults to memory/memos)
            batch: Nobody waits for this curation (the nightly run), so it may take the half-price
                batch API and an answer that is up to `ai.memory_curator.batch_wait_minutes` away;
                the sections of the condensing after it go the same way, a batch each
        """
        self.batch = batch
        self.watchlist_dir = watchlist_dir
        self.procedures_dir = procedures_dir
        self.memos_dir = memos_dir
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
    def run_condense(cls, cut: int = None, history=None, curator: 'LongTermMemoryCurator' = None) -> dict | None:
        """The condensing alone on the live bank (`//memory_condense`): one pass asked for `cut`
        words, or without it the nightly rule (`_condense`). The bank and a journal are saved as
        after a curation and the completion left in `history`; under the curation lock like a
        run. Returns the outcome, None when another curation held the lock or nothing was
        removed (the bank is then untouched)."""
        if not cls._curation_lock.acquire(blocking=False):
            info("Rem: already dreaming — skipping the condensing pass.")
            return None
        try:
            curator = curator or cls()
            bank = curator._load_memory()
            condensed, outcome = curator._condense(bank, cut)
            if condensed is bank:
                return None
            condensed = redact_for_storage(condensed)
            metadata = {'old_memory': bank, 'old_memory_size': len(bank), 'new_memory_size': len(condensed),
                        'condense': outcome, 'llm_duration_seconds': outcome['llm_duration_seconds'],
                        'timestamp': time.time()}
            curator.save_memory(condensed, metadata)
            curator.save_debug_log(condensed, metadata)
            info(f"Rem: condensing pass complete — memory bank now {len(condensed)} bytes.")
            if history is not None:
                history.add_context_message("system", cls.condense_note(outcome), time.time(), source=HISTORY_SOURCE)
            return outcome
        finally:
            cls._curation_lock.release()

    @staticmethod
    def condense_note(outcome: dict) -> str:
        """The system row a condensing pass run on its own leaves in the dialog history."""
        return f"Long term memory condensed from {outcome['from']} to {outcome['to']} words."

    @classmethod
    def last_run_at(cls) -> datetime | None:
        """When the bank was last curated, read from the newest dream journal's name
        (`save_debug_log` writes `<YYYYMMDD_HHMMSS>_<guid>_memory.md`): the one record of a run
        that survives a restart, so the nightly trigger does not depend on the process that fired
        it still being alive. None when no journal exists."""
        stamps = []
        for path in MEMORY_LOG_DIR.glob("*_memory.md"):
            try:
                stamps.append(datetime.strptime(path.name[:15], "%Y%m%d_%H%M%S"))
            except ValueError:
                continue
        return max(stamps) if stamps else None

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
                more_follow = index < total
                if more_follow != cls._backfilling:
                    cls._backfilling = more_follow
                    info(f"Rem: chunk {index}/{total} — " + ("the watch list is a past day's until the last chunk"
                                                             if more_follow else "the last chunk, the watch list is today's again"))
                try:
                    metadata = cls._curate_period(curator, period_start, period_end, history, record=total == 1,
                                                  more_follow=more_follow)
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
            cls._backfilling = False
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
                       record: bool = True, more_follow: bool = False) -> dict:
        """One period, lock already held: curate, run the commands, save the bank and its debug
        log and, when `record`, leave the completion in the history. The commands run before
        the bank is written: the bank keeps only a pointer to what the commands move into a
        procedure or a memo, so a refused `save_procedure` or `save_memo` must be known before
        the bank is replaced — its text is then kept in the bank. A period with no interactions changes nothing."""
        new_memory, metadata = curator.curate_memory(period_start, period_end, more_follow=more_follow)
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
                      lookback_days: int = 1,
                      lookahead_days: int = 5,
                      more_follow: bool = False):
        """
        Run memory curation over the interactions of a period.

        Args:
            period_start: First moment of the period (default: midnight `lookback_days` before period_end)
            period_end: Last moment of the period, exclusive (default: now)
            lookback_days: How many days back the default period_start reaches
            lookahead_days: How many days of calendar to include (default: 5)
            more_follow: The period is a chunk of a span and later chunks cover the days after it

        A bare date means midnight at its start. Both ends are accepted as dates or datetimes so a
        nightly run says "since yesterday's midnight" and a first-run curation says "since 14:03".

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
                                      period=(period_start, period_end), more_follow=more_follow)

    @staticmethod
    def word_count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def needs_condensing(word_count: int) -> bool:
        return word_count > MEMORY_WORD_LIMIT * (1 + CONDENSE_TRIGGER)

    @staticmethod
    def condense_cut(word_count: int) -> int:
        """The words a condensing pass is asked to remove: everything over the aim, CONDENSE_AIM
        under the limit — more than the excess, since a pass delivers a part of its ask."""
        return max(0, word_count - int(MEMORY_WORD_LIMIT * (1 - CONDENSE_AIM)))

    def _condense(self, bank: str, cut: int = None) -> tuple[str, dict]:
        """The condensing (`MemoryCondenser`): the bank back shorter, with what happened for the
        journal. Without `cut` the nightly rule: a pass asked for `condense_cut`, and again
        while the bank is over the trigger, CONDENSE_PASSES at most. With `cut` one pass asked
        for that many words. The bank each pass got is copied to log/memory/. `bank` itself
        comes back when nothing was removed. In a run that takes the batch API (`_batches`) each
        section is a batch of its own; one condenser for all the passes, so the first batch
        without an answer is the last one of the night."""
        start = self.word_count(bank)
        current, passes = bank, []
        condenser = MemoryCondenser(self.llm, MEMORY_WORD_LIMIT, batch=self._batches())
        for number in range(1, CONDENSE_PASSES + 1):
            words = self.word_count(current)
            asked = self.condense_cut(words) if cut is None else cut
            info(f"Rem: condensing pass {number} on {words} words (limit {MEMORY_WORD_LIMIT}), "
                 f"asked to remove {asked} words")
            self._keep_before_pass(current, number)
            condensed, outcome = condenser.condense(current, asked)
            passes.append(outcome)
            removed = words - self.word_count(condensed)
            if removed > 0:
                current = condensed
            else:
                info(f"Rem: condensing pass {number} removed nothing — the bank stays as it was")
            left = self.word_count(current)
            if cut is not None:
                trace("Rem: a pass with a given cut is not repeated")
                break
            if not self.needs_condensing(left):
                trace(f"Rem: {left} words, within the condense trigger — no further pass")
                break
            if removed < asked * CONDENSE_MIN_DELIVERY:
                error(f"Rem: condensing pass {number} removed {max(removed, 0)} of the {asked} words asked, under "
                      f"{CONDENSE_MIN_DELIVERY:.0%} — no further pass, the bank stays at {left} words "
                      f"(limit {MEMORY_WORD_LIMIT})")
                break
            if number == CONDENSE_PASSES:
                error(f"Rem: the bank is still {left} words after {CONDENSE_PASSES} condensing passes "
                      f"(limit {MEMORY_WORD_LIMIT})")
        to = self.word_count(current)
        outcome = {'from': start, 'cut': passes[0]['cut'], 'to': to, 'passes': passes,
                   'llm_duration_seconds': sum(p['llm_duration_seconds'] for p in passes),
                   'usage': "; ".join(f"pass {n}: {p['usage']}" for n, p in enumerate(passes, 1))}
        info(f"Rem: condensed to {to} words (was {start}) in {len(passes)} pass(es)"
             + (" — still over the limit" if to > MEMORY_WORD_LIMIT else ""))
        return current, outcome

    @staticmethod
    def _keep_before_pass(bank: str, number: int):
        """The bank as a condensing pass got it, copied to log/memory/: the journal diffs the
        bank as loaded against the bank as saved, so what a pass dropped shows only against
        this copy."""
        try:
            MEMORY_LOG_DIR.mkdir(parents=True, exist_ok=True)
            path = MEMORY_LOG_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_before_condense_pass{number}.md"
            path.write_text(redact_for_storage(bank))
            trace(f"Rem: the bank before condensing pass {number} kept in {path}")
        except Exception as e:
            error(f"Rem: could not keep the bank before condensing pass {number}: {e}")

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
                          period: tuple[datetime, datetime] | None = None,
                          more_follow: bool = False):
        """
        Internal method to run curation with provided data.

        Args:
            interaction_history: Recent interactions text
            calendar_data: Calendar events text
            current_memory: Current memory bank text
            period: (start, end) the history was taken from; None describes it as "the recent interactions"
            more_follow: later runs curate the days after the period (FOLLOWS_NOTE)

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
        prompt = PROMPT.replace("{WATCHLIST}", WatchList.render(self.watchlist_dir)) \
            .replace("{PROCEDURES}", Procedures.render_index(self.procedures_dir)) \
            .replace("{MEMOS}", self._memos_in_full()) \
            .replace("{MEMO_MIN_WORDS}", str(MEMO_MIN_WORDS)) \
            .replace("{MEMO_MAX_CHARS}", str(Memos.MAX_FILE_CHARS)) \
            .replace("{TODAY}", f"{datetime.now():%Y-%m-%d}") \
            .replace("{INTERACTION_HISTORY}", interaction_history) \
            .replace("{CALENDAR}", calendar_data) \
            .replace("{WATCH_WORD_LIMIT}", str(WatchList.NOTE_WORD_LIMIT)) \
            .replace("{PROCEDURE_WHEN_CHARS}", str(Procedures.WHEN_CHARS)) \
            .replace("{PERIOD}", period_text) \
            .replace("{FOLLOWS}", FOLLOWS_NOTE if more_follow else "") \
            .replace("{STT_NOTE}", STT_NOTE if voice_enabled else "") \
            .replace("{TIPS}", self._tips_prompt()) \
            .replace("{IN_PROMPT}", self._in_prompt())
        full_prompt = prompt.replace("{CURRENT_MEMORY}", current_memory)

        # Send to LLM with thinking enabled for complex reasoning
        info("Rem: deep in REM — consolidating the period's memories with the AI...")
        start_time = time.time()
        response = self._think(full_prompt, "Please curate the memory now.")
        llm_duration = time.time() - start_time
        trace(f"Rem: woke after {llm_duration:.1f}s of dreaming")
        llm_call = LLMUsage.describe_call(self.llm)
        info(f"Rem: the curation call — {llm_call}")

        response, commands = self._split_commands(self._answer(response))
        new_memory, truncated = self._accept(response, current_memory)
        condense = None
        if new_memory is current_memory:
            commands = ""
        elif self.needs_condensing(self.word_count(new_memory)):
            info(f"Rem: the curated bank is more than {CONDENSE_TRIGGER:.0%} over the limit — condensing it")
            try:
                new_memory, condense = self._condense(new_memory)
            except Exception as e:
                # The curation (a batch call of up to 90 minutes) is done and must be saved;
                # a failed pass costs only the pass, the next night's gets another go.
                error(f"Rem: the condensing pass failed, saving the curated bank uncondensed "
                      f"({self.word_count(new_memory)} words): {e!r}")
        else:
            trace("Rem: the curated bank is within the condense trigger — no condensing pass")

        # Build metadata
        metadata = {
            'period_start': period[0].isoformat(timespec='minutes') if period else '',
            'period_end': period[1].isoformat(timespec='minutes') if period else '',
            'interaction_count': interaction_count,
            'calendar_events_count': calendar_events_count,
            'old_memory_size': len(current_memory),
            'new_memory_size': len(new_memory),
            'llm_duration_seconds': llm_duration,
            'llm_call': llm_call,
            'truncated': truncated,
            'condense': condense,
            'timestamp': time.time(),
            'old_memory': current_memory,
            'full_prompt': full_prompt,
            'commands': commands,
        }

        return new_memory, metadata

    def _accept(self, response: str, current_memory: str) -> tuple[str, bool]:
        """The bank in the reply, or `current_memory` itself when the reply cannot be trusted
        (returned as the same object, so the caller can tell): a reply cut off at the output
        cap is a bank missing its tail — saving it would forget every section past the cut,
        and the next run could never bring them back — and a suspiciously short one is no
        bank at all. Also whether the reply was cut off."""
        new_memory = self._extract_memory_content(response)
        new_words = self.word_count(new_memory)
        info(f"Rem: the AI returned a {new_words}-word memory bank (limit {MEMORY_WORD_LIMIT})"
             + (" — still over the limit" if new_words > MEMORY_WORD_LIMIT else ""))
        truncated = getattr(self.llm, 'last_stop_reason', None) == 'max_tokens'
        if truncated:
            error(f"Rem: the dream was cut short — the AI hit its output limit after {len(new_memory)} chars. "
                  "Keeping the previous memory bank.")
            return current_memory, True
        if not new_memory or len(new_memory) < len(current_memory) * 0.3:
            error(f"WARNING: AI returned suspiciously short memory ({len(new_memory)} chars)")
            error("Keeping previous memory state")
            return current_memory, False
        return new_memory, False

    def _batches(self) -> bool:
        """Whether this run's calls go through the batch API at half price: it can wait
        (`batch`) and `ai.memory_curator.batch` has not turned it off."""
        if not self.batch:
            trace("Rem: a run somebody waits for, or one whose batch gave no answer — ordinary calls")
            return False
        if not ConfigManager.get_instance().get('ai.memory_curator.batch', True):
            info("Rem: ai.memory_curator.batch is off — ordinary calls")
            return False
        return True

    def _think(self, prompt: str, user_message: str) -> str:
        """The curation call: through the batch API when the run takes it (`_batches`), the
        ordinary call otherwise — and whenever the batch gave no answer (a backend without a
        batch API, the wait ran out, an error); the condensing after it then goes straight to
        ordinary calls as well. A refusal is not retried: the same request is refused again."""
        if self._batches():
            wait_minutes = float(ConfigManager.get_instance().get('ai.memory_curator.batch_wait_minutes',
                                                                  BATCH_WAIT_MINUTES))
            try:
                response = self.llm.think_batched(system_message=prompt, user_message=user_message,
                                                  thinking=Thinking.FULL, max_wait_seconds=wait_minutes * 60)
            except LLMRefusalError:
                raise
            except Exception as e:
                error(f"Rem: the batch failed ({type(e).__name__}: {e})")
                response = None
            if response is not None:
                info("Rem: the dream came back from the batch API (half price)")
                return response
            info("Rem: no answer from the batch API — making the ordinary call, and ordinary calls for the "
                 "rest of this run")
            self.batch = False
        return self.llm.think(system_message=prompt, user_message=user_message, thinking=Thinking.FULL)

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
            log_dir = MEMORY_LOG_DIR
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
                if metadata.get('llm_call'):
                    f.write(f"- Curation call: {metadata['llm_call']}\n")
                if metadata.get('truncated'):
                    f.write("- Result: reply cut off at the output cap — previous bank kept\n")
                if metadata.get('condense'):
                    c = metadata['condense']
                    f.write(f"- Condensing: {c['from']} → {c.get('to', '?')} words in {len(c.get('passes', [])) or 1} "
                            f"pass(es), {c.get('llm_duration_seconds', 0):.1f}s\n")
                    for number, p in enumerate(c.get('passes', []), 1):
                        f.write(f"- Condensing pass {number}: {p['from']} → {p['to']} words "
                                f"(asked to remove {p['cut']})\n")
                    if c.get('usage'):
                        f.write(f"- Condensing calls: {c['usage']}\n")
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
    def _block_tags(tag: str, follows: str, response: str) -> list[int]:
        """Where a block of the reply may open: each `tag` that starts its line or stands right
        after `follows`, the tag that closes what comes before the block. A tag further into a
        line is the bank writing about it."""
        pattern = rf'(?:^|{re.escape(follows)})[ \t]*({re.escape(tag)})'
        return [match.start(1) for match in re.finditer(pattern, response, re.MULTILINE)]

    @classmethod
    def _last_memory_tag(cls, response: str) -> int | None:
        """Where the last <memory> block of the reply opens, None without one. A tag inside a
        sentence counts only when no block's tag is there."""
        tags = cls._block_tags('<memory>', '</think>', response)
        if tags:
            return tags[-1]
        start = response.rfind('<memory>')
        return None if start < 0 else start

    @classmethod
    def _answer(cls, response: str) -> str:
        """The reply from its last <memory> block on — the answer itself. A local model's
        thinking can end early and arrive as the head of the reply (a bank that holds a literal
        think-close tag, quoted while thinking: ds4 on 2026-11-17), with drafts of both blocks
        in it; the bank and the commands are the last ones written. Commands drafted there are
        never run: a reply whose only <commands> block stands before its bank runs none."""
        start = cls._last_memory_tag(response)
        if start is None:
            trace("Rem: the reply has no <memory> block — reading it whole")
            return response
        head, answer = response[:start], response[start:]
        if '<memory>' in head or '<commands>' in head:
            info(f"Rem: {len(head)} chars before the reply's last <memory> block hold a block of their own "
                 "— thinking that arrived as the reply, left out")
        else:
            trace(f"Rem: the reply's <memory> block opens at char {start}")
        if '<commands>' in head and '<commands>' not in answer:
            error("Rem: the reply's only <commands> block stands before its bank — a draft or a reply "
                  "in the wrong order, its commands are NOT run")
        return answer

    @classmethod
    def _split_commands(cls, response: str) -> tuple[str, str]:
        """The reply without its last <commands> block, and the block's content ('' when there
        is none: a bank that names the tag holds no block). An unclosed block runs to the end
        of the reply."""
        tags = cls._block_tags('<commands>', '</memory>', response)
        if not tags:
            return response, ""
        match = re.compile(r'<commands>(.*?)(?:</commands>|\Z)', re.DOTALL).match(response, tags[-1])
        return response[:match.start()] + response[match.end():], match.group(1).strip()

    def _think_whole(self, prompt: str, what: str) -> str:
        """One LLM call whose reply must be complete: a reply cut off at the output cap raises."""
        response = self.llm.think(system_message=prompt, user_message="Please do it now.", thinking=Thinking.FULL)
        info(f"Memory split: the call for {what} — {LLMUsage.describe_call(self.llm)}")
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
        reply = self._think_whole(SPLIT_EXTRACT_PROMPT.replace("{TODAY}", today)
                                  .replace("{PROCEDURE_WHEN_CHARS}", str(Procedures.WHEN_CHARS))
                                  .replace("{BANK}", bank),
                                  "extracting the watch items and procedures")
        commands = self._split_commands(reply)[1]
        # A refused procedure stays in the old bank's text, which the rewrite below keeps verbatim
        results, _ = self.run_commands(commands, watchlist_dir, procedures_dir, log_dir=target / "replaced",
                                       memos_dir=target / "memos")
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
                  ("procedures", self.procedures_dir or Procedures.DIR))
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

    @staticmethod
    def _in_prompt() -> str:
        """What the AI's prompt holds live, for the curator to keep out of the bank: the data
        sections of the integrations that are on, the vault's list and the toolkits on offer.
        The curator sees none of that prompt, so without the list it cannot know what the
        bank repeats."""
        from ..commands import Toolkits
        from ..integration.IntegrationMonitor import IntegrationMonitor
        from ..prompt.PromptManager import PromptManager
        monitor = IntegrationMonitor.get_instance()
        sections, toolkits = PromptManager.data_headings(monitor), Toolkits.names(monitor)
        trace(f"Rem: the prompt is told what the AI already has: sections {', '.join(sections) or 'none'}; "
              f"{len(toolkits)} toolkit(s)")
        lines = ["## Already in the AI's prompt (FYI):",
                 "The AI has all of this live in every conversation. Do not repeat it in the bank:"]
        if sections:
            lines.append(f"- sections: {', '.join(sections)}")
        lines.append("- the vault: every `{{secret:…}}` placeholder the bank does not use is listed there with its description")
        if toolkits:
            lines.append("- toolkits it loads when needed, each with its commands and live data: " + ", ".join(toolkits))
        return "\n".join(lines)

    def _memos_in_full(self) -> str:
        """Every memo as the AI would load it: the curator cannot keep one current unread."""
        names = Memos.names(self.memos_dir)
        text = "\n\n".join(Memos.load(name, self.memos_dir) for name in names)
        info(f"Rem: the prompt carries {len(names)} memo(s) in full, {self.word_count(text)} words")
        return text or Memos.EMPTY

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
                     log_dir: Path = None, memos_dir: Path = None) -> tuple[list[str], list[str]]:
        """Run the curator's command block against the watch list, the procedures and the memos, one
        command after the other; a line that is not one of CURATOR_COMMANDS is skipped. A
        `save_watch` here is a rewrite, not a look, so it leaves the item's `checked` stamp alone.
        Returns one line per command for the debug log (the command, and why it failed if it
        did), and the bank sections that keep the text of every refused `save_procedure` and
        `save_memo`: the bank is rewritten to hold a one-line pointer in its place, so the text
        would be lost otherwise. The directories default to the curator's own; `propose_split` points them at
        its proposal."""
        watchlist_dir = watchlist_dir or self.watchlist_dir
        procedures_dir = procedures_dir or self.procedures_dir
        memos_dir = memos_dir or self.memos_dir
        if not block.strip():
            trace("Rem: no watch list, procedure or memo commands this run")
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
            elif command in Memos.COMMANDS:
                failure = Memos.run(command, params, memos_dir, log_dir)
            else:
                failure = Procedures.run(command, params, procedures_dir, log_dir, allow_replace=False)
            if failure:
                error(f"Rem: {shown} — {failure}")
                results.append(f"FAILED {shown} — {failure}")
                if command in MOVES and len(params) > 2 and "".join(params[2:]).strip():
                    unsaved.append(f"## {MOVES[command]} not saved: {params[0]}\n"
                                   f"({failure}; move it with the commands)\nWhen: {params[1]}\n\n"
                                   + ", ".join(params[2:]).strip())
                    info(f"Rem: the text of '{params[0]}' stays in the bank until it is saved")
            else:
                info(f"Rem: {shown}")
                results.append(shown)
        info(f"Rem: {len(results)} watch list / procedure / memo command(s) handled, "
             f"{sum(r.startswith('FAILED') for r in results)} failed")
        return results, unsaved

    @classmethod
    def _extract_memory_content(cls, response: str) -> str:
        """The bank in the AI's reply: its last <memory> block, up to the last closing tag (a
        bank may write about the tags) or, unclosed, to the end; without the tags a markdown
        code block, else the reply as it is."""
        start = cls._last_memory_tag(response)
        if start is not None:
            body = response[start + len('<memory>'):]
            end = body.rfind('</memory>')
            return (body if end < 0 else body[:end]).strip()
        # Try to extract from markdown code block
        match = re.search(r'```(?:markdown)?\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Otherwise use the response as-is (strip leading/trailing whitespace)
        return response.strip()
