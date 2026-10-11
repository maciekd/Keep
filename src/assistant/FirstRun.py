"""The first-time user experience (FTUE): while the install is being set up, the assistant
speaks first, has the setup toolkit at hand and sees the integrations' state in its prompt;
once the first feeds are in, a pipeline builds the first memory bank and then indexes the
photo library in full.

Setup is in progress (`in_ftue`) while fewer than `FTUE_INTEGRATION_THRESHOLD` integrations
are enabled, or while the install is younger than `FTUE_INSTALL_DAYS` — the pipeline's state
file dates it — and a command that changes the setup (`config_set`, `config_request`,
`enable_integration`, `restart_integration`, a sign-in — every command on the config lane,
see `Commands.background_command` and `note_setup_activity`; `config_help` and the other
read-only ones do not count) ran in the last `FTUE_SETUP_GRACE_HOURS`. The count alone is no measure: the first beta install had six by
the second hour and was still mid-setup, so the first week keeps setup open as long as the
user keeps at it; and a week-old install that has gone quiet, or a veteran adding one more
integration, is not being set up — neither pays for the setup toolkit. While setup is in
progress every request has the `setup` toolkit preloaded (`preload_toolkits`, through
`RequestProcessor._preload_toolkits`) so no round is spent on `load_toolkit` for it, and the
prompt carries the INTEGRATIONS section (`prompt/sections/IntegrationsSectionGenerator`):
what is on, what is on but unconfigured or broken — state that would otherwise live only in
a `list_integrations` result the history window trims away.

The brief opens the conversation on the count alone (`brief_due`): a new checkout and the
`//wipe` reset both leave no integration enabled, and the brief walks the user through the
first three. The launcher then opens the owner's chat with it as a system turn
(`ChatHub.open_conversation`) so the assistant's reply, the introduction and the next setup
step, is on screen before the user types anything.

The pipeline (`plans/FTUE_PIPELINE_PLAN.md`) is armed only on a fresh install — `begin_if_fresh`
writes `first_run.json` when the memory stub was just seeded or the bank still carries the
stub's placeholder heading; an established install has no state file and the pipeline never
runs. Its stages: SETUP (waiting for Photos to be enabled) → INDEXING (waiting for the enabled
feeds among iMessage and Photos to finish their first catch-up) → CURATING (the first bank,
`LongTermMemoryCurator.run_span` from the Photos horizon to today, chunk progress persisted so
a restart resumes, and published as `curation_progress()` for the Home tab's bar) → EXPANDING (the Photos horizon widened to `FULL_PHOTOS_BACKFILL_MONTHS`
and the feed restarted, since every photo description carries the bank) → DONE. The nightly
curation waits until the pipeline has curated once (`curation_pending`).
"""
import json
import os
import threading
import time
from dataclasses import dataclass, asdict, replace
from datetime import date, datetime
from enum import Enum
from pathlib import Path

from common.ConfigManager import config
from common.FloxyDatabase import FloxyDatabase, PHOTO_HISTORY_SOURCE
from common.Paths import AISERVER_DIR, CACHE_DIR
from common.logging import trace, info, note, error
from assistant import MemoryFile

# Below this many enabled integrations setup is in progress and the brief opens the
# conversation: the brief's three, or any three the user picked instead.
FTUE_INTEGRATION_THRESHOLD = 3

# With the count reached, setup is still in progress while the install is younger than this...
FTUE_INSTALL_DAYS = 7
# ...and a setup command ran within this long.
FTUE_SETUP_GRACE_HOURS = 48
# When the last setup command ran — `note_setup_activity` writes it, `in_ftue` reads it.
SETUP_ACTIVITY_FILE = CACHE_DIR / "setup_activity.json"

# Put in the LLM's window before every request while the FTUE lasts.
FTUE_TOOLKITS = ("setup",)

# The ConfigManager's fallback when owner_name is not set.
_NO_OWNER = "user"

BRIEF = """\
The server has just started and this install is still being set up. {owner}

Introduce yourself! This is our FTUE, be nice and engage the user, the goal is to keep the user
from dropping out and have them configure the most important integrations. Once basic integrations
are in the system starts to shine - before that we're a blank slate and provide no value for the user. 
  
First introduce yourself. Then explain the setup process briefly. First setup takes normally around 15-20 minutes. 
Indexing messages and photos gives you deep contextual knowledge of the user — that is what makes you truly smart and helpful.

Guide the user through the setup of one integration at a time, with list_integrations and config_help as your map.

Set up in this order:

1. iMessage, the feed only (no sending yet)

2. iCloud (needed so that Photos below can download offline photos)

3. Gemini API key
It's possible for the user to use the existing Anthropic API key for image processing but it will be ridiculously 
'expensive (50.000 photos -> $2000 on Opus!). Gemini 3.8 flash is FAST, ACCURATE and CHEAP for photo/video processing. 
So steer the user towards setup of a Gemini API key. A user that takes a lot of photos will have 5000 photos per year 
at average. The photo backfill will be three years. Give the user a price estimate for full index.
Check the current Gemini 3.8 Flash prices on the web (https://ai.google.dev/gemini-api/docs/pricing). Token count 
estimation per indexed asset:
- 3,000 input tokens, of which about 1,900 are served from the implicit cache for photos (none for videos) — the
  prompt carries the memory bank and the instructions beside the image, so it is far more than the image alone
- 750 output tokens, so output is normally three quarters of the bill; never price all tokens at the input rate
Prefer config_request (the masked prompt) for the key itself — don't ask the user to paste it into the chat. A key you
created yourself in the browser and can read off the page goes in with config_set.
 
3. Photos

Once you have a deep understanding of the user and his/her family/friends you will be able to be truly helpful and
be truely intelligent. iMessage and Photos are important because you'll get loads of background/context on the user
that way. 

Using Photos, iMessage and other Feeds you'll create a comprehensive long term memory of the users life. That's when
you start being able to be truly helpful and create value. 

Then whatever they want.

The user needs to know what each step does and what you need from them, not
the mechanics: config_help may hand you file paths, database names, permission systems and remote-access caveats -
translate those into plain language ("the Mac may ask you to allow this once, so it's easiest if you're sitting at it")
rather than repeating them. 

In your first message feel free to use bold/italics so the message looks nice (you can always use it but it's nice to 
start with a bit more fancy message). Feel free to use a bit of humor. Examples of whats funny during FTUE:
"right now I'm about as useful as a very polite brick"
"so functionally I'm a very charming paperweight right now"

Try to be very concise, users will churn if you write a lot of text. Keep it short. 

Rough outline of the setup process:
* We load the iMessage history and index one month of photos.
* Then curate the first long-term memory from that month of messages and photos. It will be good but not perfect; it will improve over time.
* Backfill three years of photos. Their descriptions come out better thanks to the memory above.

If the history shows you have met before, skip the introduction and continue where you left off."""

_OWNER_KNOWN = "You are talking to {name}."
_OWNER_UNKNOWN = ('owner_name is not set: ask who you are talking to and save the name with '
                  'config_set("owner_name", ...) before anything else.')


def brief_due(verbose: bool = True) -> bool:
    """Whether the first-run brief opens the conversation: fewer than
    `FTUE_INTEGRATION_THRESHOLD` integrations enabled. `verbose=False` is for a poller that
    logs the verdict itself when it changes."""
    enabled = config.enabled_integrations()
    due = len(enabled) < FTUE_INTEGRATION_THRESHOLD
    if verbose:
        trace(f"Brief check: {len(enabled)} of {FTUE_INTEGRATION_THRESHOLD} integration(s) enabled — "
              f"the brief {'opens' if due else 'is retired'}")
    return due


def install_age_days(now: float = None, state_path: Path = None) -> float | None:
    """How many days ago this install first started, by the pipeline's state file — None for an
    established install that pre-dates the pipeline and has no file."""
    state = State.load(state_path)
    if state is None:
        return None
    return ((now if now is not None else time.time()) - state.started_at) / 86400


def in_ftue(now: float = None, state_path: Path = None, activity_path: Path = None, verbose: bool = True) -> bool:
    """Whether setup is in progress: fewer than `FTUE_INTEGRATION_THRESHOLD` integrations
    enabled, or the install younger than `FTUE_INSTALL_DAYS` (an established install that
    pre-dates the pipeline has no state file and counts as old) with a setup command in the
    last `FTUE_SETUP_GRACE_HOURS`. `verbose=False` silences the trace lines, for a poller
    that logs the verdict itself when it changes."""
    log = trace if verbose else (lambda _message: None)
    if brief_due(verbose):
        log("FTUE check: too few integrations enabled — setup in progress")
        return True
    now = now if now is not None else time.time()
    age_days = install_age_days(now, state_path)
    age = f"{age_days:.1f} day(s) old" if age_days is not None else "established"
    if age_days is None or age_days >= FTUE_INSTALL_DAYS:
        log(f"FTUE check: the install is {age} (>= {FTUE_INSTALL_DAYS} days) — setup is not in progress")
        return False
    last = last_setup_activity(activity_path)
    hours_ago = (now - last) / 3600 if last is not None else None
    if hours_ago is not None and hours_ago < FTUE_SETUP_GRACE_HOURS:
        log(f"FTUE check: the install is {age} and a setup command ran {hours_ago:.1f} h ago "
              f"(< {FTUE_SETUP_GRACE_HOURS} h) — setup in progress")
        return True
    log(f"FTUE check: the install is {age} but no setup command ran in the last "
          + (f"{hours_ago:.0f} h" if hours_ago is not None else "ever recorded") + " — setup is not in progress")
    return False


def note_setup_activity(now: float = None, path: Path = None):
    """Record that a command changing the setup ran, so `in_ftue` holds for
    `FTUE_SETUP_GRACE_HOURS` more while the install is young. A failed write is logged and
    costs only this stamp."""
    path = path or SETUP_ACTIVITY_FILE
    now = now if now is not None else time.time()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"last_setup_command_at": now}))
        trace(f"Setup activity noted in {path.name}")
    except OSError as e:
        error(f"Setup activity could not be noted in {path}: {e}")


def last_setup_activity(path: Path = None) -> float | None:
    """When the last setup command ran, None when none was recorded."""
    path = path or SETUP_ACTIVITY_FILE
    try:
        return float(json.loads(path.read_text())["last_setup_command_at"])
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError, KeyError) as e:
        error(f"{path.name} is unreadable ({e}) — treating it as no setup activity")
        return None


def preload_toolkits() -> tuple[str, ...]:
    """The toolkits every request preloads on top of its channel's own: the setup toolkit
    while setup is in progress, nothing after."""
    return FTUE_TOOLKITS if in_ftue() else ()


def brief() -> str:
    """The system turn that opens the first conversation."""
    name = config.get_owner_name()
    owner = _OWNER_UNKNOWN if name == _NO_OWNER else _OWNER_KNOWN.format(name=name)
    return BRIEF.format(owner=owner)


# --- The pipeline -------------------------------------------------------------------------

STATE_FILE = AISERVER_DIR / "first_run.json"
PLACEHOLDER_HEADING = MemoryFile.PLACEHOLDER_HEADING
# The Photos horizon once the first bank exists: what config.example.yaml's one month becomes.
FULL_PHOTOS_BACKFILL_MONTHS = 36
PHOTOS_BACKFILL_KEY = "feeds.photos.backfill_months"
PIPELINE_POLL_SECONDS = 2     # a tick is a file read and a few cached lookups; the waits log only on change
CURATION_RETRY_SECONDS = 600  # how long an all-failed first curation waits before it is run again
CURATION_STEP_SECONDS = 160.0   # what one curation chunk normally takes: the bar's estimate until one has finished
PHOTOS_INTEGRATION = "Photos"
IMESSAGE_INTEGRATION = "iMessage"
PHOTOS_FEED = PHOTO_HISTORY_SOURCE
IMESSAGE_FEED = "iMessage"
HISTORY_SOURCE = "first_run"     # the `source` of the system rows the pipeline leaves


class Stage(str, Enum):
    SETUP = "setup"             # integrations being enabled; waiting for Photos
    INDEXING = "indexing"       # the first catch-up of the feeds (one month of photos, iMessage)
    CURATING = "curating"       # the first memory bank, chunk by chunk
    EXPANDING = "expanding"     # the Photos horizon widened, the feed reading the rest
    DONE = "done"


@dataclass
class State:
    stage: Stage
    started_at: float
    entered_at: float                    # when the current stage began
    curation_start: str | None = None    # the first curation's span, ISO dates, fixed when CURATING begins
    curation_end: str | None = None
    chunks_done: int = 0
    chunks_total: int = 0

    @classmethod
    def fresh(cls, now: float = None) -> 'State':
        now = now if now is not None else time.time()
        return cls(stage=Stage.SETUP, started_at=now, entered_at=now)

    def enter(self, stage: Stage, now: float = None):
        self.stage = stage
        self.entered_at = now if now is not None else time.time()

    @property
    def span(self) -> tuple[date, date] | None:
        if not self.curation_start or not self.curation_end:
            return None
        return date.fromisoformat(self.curation_start), date.fromisoformat(self.curation_end)

    def save(self, path: Path = None):
        path = path or STATE_FILE
        data = asdict(self)
        data['stage'] = self.stage.value
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, path)
        trace(f"First run: state saved — {self.stage.value}")

    @classmethod
    def load(cls, path: Path = None) -> 'State | None':
        """The persisted state, or None when this install never armed the pipeline."""
        path = path or STATE_FILE
        try:
            data = json.loads(path.read_text())
            data['stage'] = Stage(data['stage'])
            return cls(**data)
        except FileNotFoundError:
            return None     # never armed, or deleted (//wipe) between two of the pipeline's polls
        except (ValueError, TypeError, KeyError) as e:
            error(f"First run: {path.name} is unreadable ({e}) — treating the install as established")
            return None


def begin_if_fresh(seeded: bool, path: Path = None) -> bool:
    """Arm the pipeline on a fresh install: the memory stub was just seeded, or the bank still
    opens with the stub's placeholder heading (seeded before the pipeline existed). An install
    with a state file keeps it; an established install gets none, and the pipeline never
    touches it. Returns whether the pipeline is armed."""
    path = path or STATE_FILE
    if path.exists():
        state = State.load(path)
        info(f"First run: pipeline state found — {state.stage.value if state else 'unreadable'}")
        return state is not None
    if not seeded and not MemoryFile.is_placeholder(MemoryFile.read()):
        info("First run: established install (a curated memory bank, no pipeline state) — no pipeline")
        return False
    State.fresh().save(path)
    note("First run: fresh install — the first-run pipeline is armed (waiting for Photos to be enabled)")
    return True


def stage(path: Path = None) -> Stage | None:
    """The pipeline's stage, None on an established install."""
    state = State.load(path)
    return state.stage if state else None


def curation_pending(path: Path = None) -> bool:
    """Whether the first curation is still ahead: the nightly run waits for it, so two days
    of curation do not erode the stub before the real first pass."""
    return stage(path) in (Stage.SETUP, Stage.INDEXING)


@dataclass(frozen=True)
class CurationProgress:
    """The first curation as it runs, for the Home tab's bar. `fraction()` moves smoothly
    through the current chunk by extrapolating from the chunks this process has finished —
    the average of their times, `CURATION_STEP_SECONDS` until one has — and never claims
    more than the chunk boundary a slow chunk is heading for. Times are `time.monotonic()`."""
    chunks_total: int
    chunks_done: int            # in the span, the chunks a previous process did included
    started_at: float           # when this process started curating
    resumed_at: int             # chunks_done when it started
    done_at: float              # when the last chunk finished; started_at until one has

    @classmethod
    def begin(cls, chunks_total: int, chunks_done: int, now: float = None) -> 'CurationProgress':
        now = time.monotonic() if now is None else now
        return cls(chunks_total, chunks_done, now, chunks_done, now)

    def chunk_done(self, index: int, now: float = None) -> 'CurationProgress':
        return replace(self, chunks_done=index, done_at=time.monotonic() if now is None else now)

    @property
    def step_seconds(self) -> float:
        finished_here = self.chunks_done - self.resumed_at
        if finished_here <= 0:
            return CURATION_STEP_SECONDS
        return (self.done_at - self.started_at) / finished_here

    def _current(self, now: float) -> float:
        """How far through the current chunk, 0..1, on the step estimate."""
        step = self.step_seconds
        return min(1.0, (now - self.done_at) / step) if step > 0 else 1.0

    def fraction(self, now: float = None) -> float:
        if self.chunks_total <= 0:
            return 0.0
        now = time.monotonic() if now is None else now
        return min(1.0, (self.chunks_done + self._current(now)) / self.chunks_total)

    def eta_seconds(self, now: float = None) -> float:
        """The chunks left at the step estimate, the current one's elapsed time taken off."""
        now = time.monotonic() if now is None else now
        remaining = max(0, self.chunks_total - self.chunks_done) - self._current(now)
        return max(0.0, remaining * self.step_seconds)


def curation_progress() -> CurationProgress | None:
    """The first curation while the pipeline runs it, None otherwise — the nightly run and
    `//memory_curate` are not shown."""
    return Pipeline.curation


def _enabled(name: str) -> bool:
    return name.lower() in (n.lower() for n in config.enabled_integrations())


def _day(d: date) -> str:
    """15 August — the history rows are read by the family, not by a parser."""
    return f"{d.day} {d:%B}"


class Pipeline:
    """The thread that moves the state through its stages, polling every `poll_seconds`.
    `feeds` answers `runner(name)` (default: the process-wide `Feeds`); `history` is the
    assistant's MessageHistory for the system rows and the curation completions. Ends on its
    own once the stage is DONE, or at once on an established install."""

    curation: CurationProgress | None = None    # the running first curation, for `curation_progress()`

    def __init__(self, history, poll_seconds: float = PIPELINE_POLL_SECONDS, feeds=None, path: Path = None):
        self._history = history
        self._poll_seconds = poll_seconds
        self._feeds = feeds
        self._path = path or STATE_FILE
        self._stop_event = threading.Event()
        self._waiting_on: str | None = None     # the last wait reason traced, so a poll logs only a change
        self._retry_at: float = 0.0             # monotonic: when an all-failed first curation may run again
        self._thread = threading.Thread(target=self._run, name="first-run-pipeline", daemon=True)

    def start(self):
        if State.load(self._path) is None:
            info("First run: no pipeline state — the pipeline does not start")
            return
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def _run(self):
        """Polls until stopped. A missing state file or a DONE stage leaves nothing to do but
        the thread keeps polling: `//first_run stage <name>` must be able to set it going
        again, and the idle poll costs nothing."""
        note("First run: pipeline started")
        idle_reported = None
        while not self._stop_event.is_set():
            state = State.load(self._path)
            idle = "state gone" if state is None else ("done" if state.stage == Stage.DONE else None)
            if idle:
                if idle != idle_reported:
                    info(f"First run: pipeline {idle} — idle until the state changes (//first_run stage <name>)")
                    idle_reported = idle
            else:
                idle_reported = None
                try:
                    if self.tick(state):
                        continue    # a stage just began: give it its first tick at once
                except Exception as e:
                    error(f"First run: {state.stage.value} step failed: {type(e).__name__}: {e}")
            self._stop_event.wait(self._poll_seconds)

    # --- one step ---------------------------------------------------------------------------

    def tick(self, state: State) -> bool:
        """One poll: advance the stage when its condition holds. Returns whether it advanced."""
        step = {Stage.SETUP: self._setup, Stage.INDEXING: self._indexing,
                Stage.CURATING: self._curating, Stage.EXPANDING: self._expanding}.get(state.stage)
        return bool(step and step(state))

    def _setup(self, state: State) -> bool:
        if not _enabled(PHOTOS_INTEGRATION):
            self._waiting("waiting for the Photos integration to be enabled")
            return False
        self._advance(state, Stage.INDEXING, "Photos is enabled: waiting for the first catch-up of the feeds "
                                              "(one month of photos, the iMessage history)",
                      row="Photos is set up. Reading the last month of photos and the message history, "
                          "the groundwork for a long-term memory of the household.")
        return True

    def _required_feeds(self) -> list[str]:
        feeds = [PHOTOS_FEED]
        if _enabled(IMESSAGE_INTEGRATION):
            feeds.append(IMESSAGE_FEED)
        return feeds

    def _indexing(self, state: State) -> bool:
        from assistant.integration.FeedRunner import FeedState, describe_feed_status
        waiting = []
        for name in self._required_feeds():
            runner = self._runner(name)
            if runner is None or runner.state != FeedState.LIVE:
                waiting.append(f"{name} ({describe_feed_status(runner.status() if runner else None)})")
        if waiting:
            self._waiting(f"still indexing — {', '.join(waiting)}")
            return False
        start = self._photos_horizon()
        end = date.today()
        from assistant.agents.LongTermMemoryCurator import LongTermMemoryCurator
        state.curation_start, state.curation_end = start.isoformat(), end.isoformat()
        state.chunks_done, state.chunks_total = 0, len(LongTermMemoryCurator.chunks(start, end))
        self._advance(state, Stage.CURATING,
                      f"The feeds are in: curating the first memory bank over {start:%Y-%m-%d} .. {end:%Y-%m-%d} "
                      f"in {state.chunks_total} chunk(s)",
                      row=f"Photos and messages are in. Building the first long-term memory from what happened "
                          f"between {_day(start)} and {_day(end)}, in {state.chunks_total} steps of a few days each.")
        return True

    def _photos_horizon(self) -> date:
        """Where the Photos catch-up started: the feeds row's horizon, else the runner's."""
        row = FloxyDatabase.get_instance().get_feed(PHOTOS_FEED)
        horizon = row.get("horizon") if row else None
        if horizon is None:
            runner = self._runner(PHOTOS_FEED)
            horizon = runner.status()["horizon_ts"] if runner else time.time()
            info(f"First run: no persisted Photos horizon, using the runner's ({horizon})")
        return datetime.fromtimestamp(horizon).date()

    def _curating(self, state: State) -> bool:
        """Runs the span, which takes hours; `//first_run reset` or `stage` meanwhile is
        honoured at the next chunk: the progress is not saved over the new state and the
        span stops. A span whose every chunk failed (an API outage) is not "built": the
        stage stays, the progress rewinds and the span is tried again after a pause."""
        from assistant.agents.LongTermMemoryCurator import LongTermMemoryCurator
        span = state.span
        if span is None:
            error("First run: CURATING without a span — skipping the first curation")
            return self._expand(state)
        if time.monotonic() < self._retry_at:
            self._waiting(f"the first curation failed; trying again in {self._retry_at - time.monotonic():.0f}s")
            return False
        resumed_at = state.chunks_done

        def on_chunk(index, total, metadata):
            current = State.load(self._path)
            if current is None or current.stage != Stage.CURATING:
                changed = "reset" if current is None else f"forced to {current.stage.value}"
                info(f"First run: the state was {changed} during the curation — stopping the span after chunk "
                     f"{index}/{total}, progress not saved")
                return False
            state.chunks_done, state.chunks_total = index, total
            state.save(self._path)
            Pipeline.curation = Pipeline.curation.chunk_done(index)
            trace(f"First run: curation chunk {index}/{total} done — {Pipeline.curation.step_seconds:.0f}s per chunk so far")
            return True

        Pipeline.curation = CurationProgress.begin(len(LongTermMemoryCurator.chunks(*span)), resumed_at)
        try:
            summary = LongTermMemoryCurator.run_span(*span, history=self._history, on_chunk=on_chunk,
                                                     first_chunk=resumed_at)
        finally:
            Pipeline.curation = None
        if summary is None:
            info("First run: another curation is running — trying again at the next poll")
            return False
        if summary.get('aborted'):
            info("First run: the first curation was stopped — the state changed underneath it")
            return False
        ran = summary['chunks'] - resumed_at
        if summary['failed'] and summary['failed'] == ran:
            error(f"First run: all {ran} chunk(s) of the first curation failed — the bank is not built; trying again "
                  f"from chunk {resumed_at + 1} in {CURATION_RETRY_SECONDS}s")
            state.chunks_done = resumed_at
            state.save(self._path)
            self._retry_at = time.monotonic() + CURATION_RETRY_SECONDS
            return False
        self._retry_at = 0.0
        return self._expand(state, summary)

    def _expand(self, state: State, summary: dict = None) -> bool:
        """Widen the Photos horizon and restart the feed; a horizon the user already set at
        least as wide is left alone, and the pipeline is then done."""
        curated = f"{summary['curated']} of {summary['chunks']} chunk(s) curated" if summary else "no curation"
        if summary is None:
            built = "The first long-term memory step was skipped."
        elif summary['failed']:
            built = f"The first long-term memory is built, though {summary['failed']} of its {summary['chunks']} steps failed."
        else:
            built = "The first long-term memory is built."
        current = config.get(PHOTOS_BACKFILL_KEY, 1)
        try:
            current = float(current)
        except (TypeError, ValueError):
            current = 1.0
        if current >= FULL_PHOTOS_BACKFILL_MONTHS:
            self._advance(state, Stage.DONE, f"first memory bank built ({curated}); the Photos horizon is already "
                                             f"{current:g} months, nothing to widen",
                          row=f"{built} The photo library was already set to reach {current:g} months back, "
                              "so there is nothing more to index.")
            return True
        from common import ConfigStore
        from assistant.integration.IntegrationMonitor import IntegrationMonitor
        ConfigStore.set_value(PHOTOS_BACKFILL_KEY, str(FULL_PHOTOS_BACKFILL_MONTHS))
        # Stamped before the restart: `_expanding` accepts a `live_since` from the stage's start on, and the
        # restart is synchronous — a small library can go live before it returns.
        entered_at = time.time()
        result = IntegrationMonitor.get_instance().restart(PHOTOS_INTEGRATION)
        if result is None:
            error(f"First run: no '{PHOTOS_INTEGRATION}' integration to restart — the wider horizon takes effect "
                  "at the next start")
        self._advance(state, Stage.EXPANDING, f"first memory bank built ({curated}); Photos horizon widened from "
                                              f"{current:g} to {FULL_PHOTOS_BACKFILL_MONTHS} months, feed restarted",
                      row=f"{built} Now indexing the last {FULL_PHOTOS_BACKFILL_MONTHS // 12} years of photos and "
                          "videos in the background; with the memory in place their descriptions come out better.",
                      entered_at=entered_at)
        return True

    def _expanding(self, state: State) -> bool:
        from assistant.integration.FeedRunner import FeedState, describe_feed_status
        runner = self._runner(PHOTOS_FEED)
        status = runner.status() if runner else None
        live = status is not None and status["state"] == FeedState.LIVE.value \
            and (status["live_since"] or 0) >= state.entered_at
        if not live:
            self._waiting(f"the wider photo index is still catching up — {describe_feed_status(status)}")
            return False
        self._advance(state, Stage.DONE, f"the photo library is indexed ({status['stored']} assets)",
                      row=f"The photo library is indexed: {status['stored']} photos and videos are searchable. "
                          "The first-run setup is complete.")
        return True

    # --- helpers ----------------------------------------------------------------------------

    def _runner(self, name: str):
        if self._feeds is None:
            from assistant.integration.FeedRunner import Feeds
            self._feeds = Feeds.get_instance()
        return self._feeds.runner(name)

    def _waiting(self, reason: str):
        """Trace why the stage has not advanced — once per reason, not once per poll."""
        if reason != self._waiting_on:
            trace(f"First run: {reason}")
            self._waiting_on = reason

    def _advance(self, state: State, stage: Stage, what: str, row: str, entered_at: float = None):
        """`what` is the log's technical line; `row` the plain-language system row the Home
        tab shows and the LLM reads. The pipeline never speaks as the assistant: a
        hard-coded first-person message has no way to carry its voice. `entered_at` backdates
        the stage to when its work really began."""
        self._waiting_on = None
        state.enter(stage, now=entered_at)
        state.save(self._path)
        note(f"First run: {what} → {stage.value}")
        if self._history is not None:
            self._history.add_context_message("system", row, time.time(), source=HISTORY_SOURCE)
