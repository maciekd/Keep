"""One thread per Feed: catch up from the persisted cursor (or the horizon, the first time),
then poll live. The cursor is written back after every stored batch, so a restart resumes
where it left off; "backfill" is simply the gap between the cursor and now
(plans/FEED_PLAN.md §3.2). The counts are the history's: `stored` is how many rows the feed
has there (counted at start, bumped per batch, recounted now and then), the catch-up's total
is stored + remaining and progress the ratio — the same numbers before and after a restart.
A horizon moved further back in the config reads from the new horizon until the cursor reaches
the one the feed had, then carries on from there. Inside the window it had already covered —
the old horizon up to the newest entry stored before the move — an entry is dropped only when
the history holds a row of this feed at its moment (live rows can carry a media description the
backfill path does not, so the history's dedup index would not catch those); a hole in the
window, and anything past it, was never stored and is kept. Once live and idle the runner also
asks the feed to `sweep` for items behind the cursor it has no row for (a batch the provider
refused, rows a migration forgot) — holes that nothing else would ever close. A batch the
history refuses is held and stored again on the next pass, never refetched: a Photos batch's
entries are the product of paid LLM calls its ledger will not repeat. `Feeds` is the
process-wide registry the bootstrap start hook, `restart_integration` and the status surfaces
go through."""
import threading
import time
from enum import Enum
from typing import Optional

from common.ConfigManager import config
from common.DevelopmentConfig import RUNNING_UNDER_TEST
from common.FloxyDatabase import FloxyDatabase
from common.logging import info, trace, error, fine, note
from ..Telemetry import Telemetry
from .Feed import Feed, FeedBatch, FEED_ROLE

SECONDS_PER_MONTH = 30.44 * 24 * 3600


class FeedState(Enum):
    STARTING = "starting"
    CATCHING_UP = "catching_up"
    LIVE = "live"
    STOPPED = "stopped"
    FAILED = "failed"


class FeedRunner:
    FAILURE_LIMIT = 5   # consecutive fetch failures before the feed gives up
    # An ETA needs a rate worth trusting: this much of the catch-up done in this process, over this long.
    ETA_MIN_ITEMS = 5
    ETA_MIN_SECONDS = 30.0
    RECOUNT_SECONDS = 60.0      # how often the feed thread re-reads its row count from the history
    SWEEP_SECONDS = 900.0       # how often a live, idle feed is asked for items behind its cursor it has no row for

    def __init__(self, feed: Feed, message_history, horizon_ts: float, db: Optional[FloxyDatabase] = None):
        self._feed = feed
        self._history = message_history
        self._horizon_ts = horizon_ts
        self._db = db
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._opened = threading.Event()    # set once open_feed has returned or raised (see `wait_opened`)
        # Cleared = paused: the loop idles between fetches. //feed pause flips it; the state
        # (catching up / live) and the cursor stay where they are, resume just carries on.
        self._running = threading.Event()
        self._running.set()
        self._thread: Optional[threading.Thread] = None
        self._state = FeedState.STARTING
        self._cursor: Optional[str] = None
        self._stored = 0             # the feed's rows in the history (source = name, role = FEED_ROLE)
        self._recounted_at: Optional[float] = None
        self._caught_up = 0          # rows stored during the current catch-up, for the go-live line
        self._remaining: Optional[int] = None
        self._last_entry_ts: Optional[float] = None
        self._live_since: Optional[float] = None
        self._live_boundary: Optional[float] = None
        self._session_remaining: Optional[int] = None    # what was left when this process began catching up
        self._rate_since: Optional[float] = None         # when _session_remaining was measured (the ETA's clock)
        self._paused_seconds = 0.0                      # pauses so far, kept out of the ETA's clock
        self._paused_at: Optional[float] = None
        # A horizon move-back in progress: entries at or past `_gap_until` (the old horizon) are
        # already in the history and are dropped; once the gap is read the cursor jumps to
        # `_resume_cursor`, where the feed was before the move-back.
        self._gap_until: Optional[float] = None
        self._resume_cursor: Optional[str] = None
        # The newest entry stored before the move-back: the covered window's far edge. None
        # (a gap opened before this was recorded) drops everything past `_gap_until`.
        self._gap_covered_until: Optional[float] = None
        self._unstored: Optional[tuple[FeedBatch, bool]] = None    # a batch the history refused, stored again next pass
        self._sweep_due: Optional[float] = None    # when the live feed is next asked to sweep; None: at the first idle poll
        self._last_error: Optional[str] = None
        self._failures = 0

    @property
    def name(self) -> str:
        return self._feed.feed_name

    @property
    def state(self) -> FeedState:
        return self._state

    @property
    def paused(self) -> bool:
        return not self._running.is_set()

    def pause(self) -> bool:
        """Stop fetching after the current batch. Returns False when already paused."""
        if self.paused:
            trace(f"Feed '{self.name}' already paused")
            return False
        self._running.clear()
        self._paused_at = time.time()
        info(f"Feed '{self.name}' paused ({self._state.value})")
        return True

    def resume(self) -> bool:
        """Carry on from the cursor. Returns False when not paused."""
        if not self.paused:
            trace(f"Feed '{self.name}' is not paused")
            return False
        self._running.set()
        if self._paused_at is not None:
            self._paused_seconds += time.time() - self._paused_at
            self._paused_at = None
        info(f"Feed '{self.name}' resumed ({self._state.value})")
        return True

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            trace(f"Feed '{self.name}' already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=f"Feed-{self.name}", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0):
        self._stop_event.set()
        self._running.set()     # a paused thread must wake up to see the stop
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                info(f"Feed '{self.name}': thread still busy after {timeout:.0f}s, not waiting further")
        with self._lock:
            if self._state not in (FeedState.FAILED,):
                self._state = FeedState.STOPPED
        try:
            self._feed.close_feed()
        except Exception as e:
            trace(f"Feed '{self.name}': close_feed raised {type(e).__name__}: {e}")

    def status(self) -> dict:
        in_flight = self._in_flight()
        now = time.time()
        with self._lock:
            stored, remaining = self._stored, self._remaining
            if self._state == FeedState.CATCHING_UP and in_flight:
                # the batch under way: what its workers have finished counts as done, not as left
                stored += in_flight
                if remaining is not None:
                    remaining = max(0, remaining - in_flight)
            return {
                "state": self._state.value,
                "paused": self.paused,
                "items": self._feed.item_noun,
                "stored": stored,
                "remaining": remaining,
                "total": stored + remaining if remaining is not None else None,
                "progress": self._progress(stored, remaining),
                "eta_seconds": self._eta(remaining, now),
                "horizon_ts": self._horizon_ts,
                "last_entry_ts": self._last_entry_ts,
                "live_since": self._live_since,
                "last_error": self._last_error,
            }

    def _in_flight(self) -> int:
        try:
            return self._feed.in_flight()
        except Exception as e:
            trace(f"Feed '{self.name}': in_flight() raised {type(e).__name__}: {e}")
            return 0

    def _eta(self, remaining: Optional[int], now: float) -> Optional[float]:
        """Seconds until the catch-up is done, from this process's own rate: what came off
        `remaining` since it started catching up, over the time it has been running unpaused.
        None until ETA_MIN_ITEMS and ETA_MIN_SECONDS are in, and for anything but a counted
        catch-up."""
        if self._state != FeedState.CATCHING_UP or remaining is None \
                or self._session_remaining is None or self._rate_since is None:
            return None
        done = self._session_remaining - remaining
        elapsed = now - self._rate_since - self._paused_seconds
        if self._paused_at is not None:
            elapsed -= now - self._paused_at
        if done < self.ETA_MIN_ITEMS or elapsed < self.ETA_MIN_SECONDS:
            return None
        return remaining * elapsed / done

    def _progress(self, stored: int, remaining: Optional[int]) -> Optional[float]:
        """0..1 through the catch-up: stored over stored + remaining when the feed can count
        what is left, else how far the newest stored entry has come between the horizon and
        the live boundary."""
        if self._state != FeedState.CATCHING_UP:
            return None
        if remaining is not None:
            total = stored + remaining
            return stored / total if total else 1.0
        if self._last_entry_ts is None or self._live_boundary is None:
            return 0.0
        span = self._live_boundary - self._horizon_ts
        if span <= 0:
            return 1.0
        return max(0.0, min(1.0, (self._last_entry_ts - self._horizon_ts) / span))

    # --- the thread ---------------------------------------------------------------------------

    def _database(self) -> FloxyDatabase:
        if self._db is None:
            self._db = FloxyDatabase.get_instance()
        return self._db

    def wait_opened(self, timeout: float) -> bool:
        """Block until the thread's open_feed has come back, either way, or `timeout` runs
        out; True when it has. A health check probed right after `start` (a restart) reads
        state the feed thread has not built yet — Telegram's client, say — and reports a
        warning that stands until the monitor's next cycle, 15 minutes later."""
        return self._opened.wait(timeout)

    def _run(self):
        try:
            opened = self._open()
            self._opened.set()
            if not opened:
                return
            self._load_cursor()
            self._loop()
        except Exception as e:
            error(f"Feed '{self.name}' crashed: {type(e).__name__}: {e}")
            with self._lock:
                self._state = FeedState.FAILED
                self._last_error = f"{type(e).__name__}: {e}"
        finally:
            self._opened.set()
            if self._unstored is not None:
                self._flush_unstored()
            if self._state not in (FeedState.FAILED,):
                with self._lock:
                    self._state = FeedState.STOPPED
            try:
                self._feed.close_feed()
            except Exception as e:
                trace(f"Feed '{self.name}': close_feed raised {type(e).__name__}: {e}")
            Telemetry.get_instance().feed_stopped(self.name)

    def _flush_unstored(self):
        """One last try at the batch the history refused before the thread ends: its entries
        can be the product of LLM calls the feed will not repeat (the Photos ledger already
        lists those photos), so losing them means history rows that never come back."""
        batch, gap_closed = self._unstored
        try:
            self._store(batch, gap_closed)
            info(f"Feed '{self.name}': the {len(batch.entries)} held entries were stored on the way out")
        except Exception as e:
            error(f"Feed '{self.name}': {len(batch.entries)} held entries could not be stored before stopping "
                  f"({type(e).__name__}: {e}); their history rows are lost")

    def _open(self) -> bool:
        try:
            opened = self._feed.open_feed()
        except Exception as e:
            error(f"Feed '{self.name}': open failed: {type(e).__name__}: {e}")
            with self._lock:
                self._state = FeedState.FAILED
                self._last_error = f"{type(e).__name__}: {e}"
            return False
        if not opened:
            reason = self._feed.open_error or "cannot open"
            info(f"Feed '{self.name}' not starting: {reason}")
            with self._lock:
                self._state = FeedState.STOPPED
                self._last_error = reason
            return False
        return True

    def _recount_stored(self):
        """Read the feed's count: its own (`Feed.count_stored`) when it keeps one, else its rows
        in the history. A failed count keeps the running one."""
        try:
            stored = self._feed.count_stored()
            if stored is None:
                stored = self._database().count_history_entries(self.name, FEED_ROLE)
        except Exception as e:
            trace(f"Feed '{self.name}': row count failed ({type(e).__name__}: {e}), keeping {self._stored}")
            return
        with self._lock:
            if stored != self._stored:
                trace(f"Feed '{self.name}': {self._stored} → {stored} rows in the history")
            self._stored = stored
            self._recounted_at = time.time()

    def _report_count(self):
        """The telemetry's count of what the feed holds; its hourly status carries the number
        of that moment for every running feed."""
        try:
            Telemetry.get_instance().feed_count(self.name, self._stored)
        except Exception as e:
            error(f"Feed '{self.name}': telemetry count failed: {type(e).__name__}: {e}")

    def _load_cursor(self):
        """Resume from the feeds row; start at the horizon when there is none. A horizon
        moved further back in the config resets the cursor there and opens a gap up to the
        old horizon, with the cursor it had as the place to jump back to (see `_trim_to_gap`).
        Moved back again while a gap is still being read, the boundary and the resume cursor
        stay: the unread rest of the earlier gap must not become a hole, and the part of it
        read twice dedupes since both passes are backfill rows."""
        self._recount_stored()
        row = self._database().get_feed(self.name)
        if row is None:
            cursor = self._feed.initial_cursor(self._horizon_ts)
            note(f"Feed '{self.name}': first run, catching up from "
                 f"{time.strftime('%Y-%m-%d', time.localtime(self._horizon_ts))} (cursor={cursor}"
                 f"{f', {self._stored} rows already in the history' if self._stored else ''})")
            self._database().save_feed(self.name, cursor=cursor, horizon=self._horizon_ts)
        else:
            cursor = row["cursor"]
            self._last_entry_ts = row["last_entry_ts"]
            self._gap_until = row["gap_until"]
            self._resume_cursor = row["resume_cursor"]
            self._gap_covered_until = row["gap_covered_until"]
            stored_horizon = row["horizon"]
            if stored_horizon is not None and self._horizon_ts < stored_horizon - 86400:
                if self._gap_until is None:
                    self._gap_until, self._resume_cursor = stored_horizon, cursor
                    # covered: the old horizon up to the newest entry stored; a feed that
                    # stored nothing yet covers nothing past the boundary
                    self._gap_covered_until = self._last_entry_ts if self._last_entry_ts is not None else stored_horizon
                cursor = self._feed.initial_cursor(self._horizon_ts)
                note(f"Feed '{self.name}': horizon moved back to "
                     f"{time.strftime('%Y-%m-%d', time.localtime(self._horizon_ts))}, reading the gap up to "
                     f"{time.strftime('%Y-%m-%d', time.localtime(self._gap_until))} "
                     f"(cursor={cursor}), then carrying on from cursor={self._resume_cursor}")
                self._database().save_feed(self.name, cursor=cursor, horizon=self._horizon_ts,
                                           gap_until=self._gap_until, resume_cursor=self._resume_cursor,
                                           gap_covered_until=self._gap_covered_until)
            else:
                if stored_horizon is not None and self._horizon_ts > stored_horizon + 86400:
                    info(f"Feed '{self.name}': the configured horizon "
                         f"({time.strftime('%Y-%m-%d', time.localtime(self._horizon_ts))}) is newer than the stored "
                         f"one ({time.strftime('%Y-%m-%d', time.localtime(stored_horizon))}); a horizon only moves "
                         f"back, keeping the stored one — a shorter backfill_months never drops rows")
                self._horizon_ts = stored_horizon if stored_horizon is not None else self._horizon_ts
                gap = (f", mid-gap up to {time.strftime('%Y-%m-%d', time.localtime(self._gap_until))}"
                       if self._gap_until is not None else "")
                info(f"Feed '{self.name}': resuming from cursor={cursor} ({self._stored} rows in the history{gap})")
        with self._lock:
            self._cursor = cursor
            self._state = FeedState.CATCHING_UP
            self._live_boundary = time.time()
            self._caught_up = 0
        self._refresh_remaining(initial=True)

    def _refresh_remaining(self, initial: bool = False):
        """Re-count what is ahead of the cursor. On `initial` the ETA's clock starts: the
        catch-up's size needs no fixing, stored + remaining is the same total before and after
        a restart."""
        try:
            remaining = self._feed.remaining(self._cursor)
            if remaining is not None and self._gap_until is not None:
                # only the gap is ahead: what lies past the resume cursor is not this catch-up's
                covered = self._feed.remaining(self._resume_cursor)
                remaining = max(0, remaining - (covered or 0))
        except Exception as e:
            trace(f"Feed '{self.name}': remaining() raised {type(e).__name__}: {e}")
            remaining = None
        with self._lock:
            self._remaining = remaining
            if initial:
                self._session_remaining = remaining
                self._rate_since = time.time()
                self._paused_seconds = 0.0
        if initial and remaining:
            info(f"Feed '{self.name}': {remaining} item(s) to catch up on ({self._stored} already stored)")
        elif initial and remaining == 0:
            trace(f"Feed '{self.name}': nothing ahead of the cursor")

    def _loop(self):
        while not self._stop_event.is_set():
            if self.paused:
                trace(f"Feed '{self.name}': paused, waiting")
                self._running.wait()
                continue
            if self._recounted_at is None or time.time() - self._recounted_at >= self.RECOUNT_SECONDS:
                self._recount_stored()
            self._report_count()
            swept = False
            try:
                if self._unstored is not None:
                    batch, gap_closed = self._unstored
                    info(f"Feed '{self.name}': storing the {len(batch.entries)} entries the history refused last pass")
                else:
                    batch = self._feed.fetch(self._cursor, backfill=self._state == FeedState.CATCHING_UP)
                    batch, gap_closed = self._trim_to_gap(batch)
                self._store(batch, gap_closed)
                self._failures = 0
                if self._state == FeedState.LIVE and batch.exhausted and not batch.entries and not gap_closed:
                    swept = self._sweep()
            except Exception as e:
                self._failures += 1
                error(f"Feed '{self.name}': fetch or store failed ({self._failures}/{self.FAILURE_LIMIT}): "
                      f"{type(e).__name__}: {e}")
                with self._lock:
                    self._last_error = f"{type(e).__name__}: {e}"
                if self._failures >= self.FAILURE_LIMIT:
                    error(f"Feed '{self.name}' giving up after {self._failures} consecutive failures; "
                          f"restart_integration brings it back")
                    with self._lock:
                        self._state = FeedState.FAILED
                    return
                self._stop_event.wait(self._feed.poll_interval)
                continue

            if gap_closed:
                # the rest of the catch-up, from the resume cursor, measures itself afresh
                self._refresh_remaining(initial=True)
                continue
            if self._state == FeedState.CATCHING_UP:
                self._refresh_remaining()
                if batch.exhausted:
                    self._go_live()
            if batch.exhausted and not swept:
                self._stop_event.wait(self._feed.poll_interval)

    def _sweep(self) -> bool:
        """Ask the feed for items behind the cursor it has no row for (`Feed.sweep`) when a
        sweep is due, and store what it brings without moving the cursor. Returns True when
        more is waiting, so the loop goes straight on instead of sleeping until the poll."""
        now = time.time()
        if self._sweep_due is not None and now < self._sweep_due:
            return False
        batch = self._feed.sweep(self._horizon_ts, self._cursor)
        if batch is None:
            self._sweep_due = float("inf")
            trace(f"Feed '{self.name}': never sweeps (the feed cannot tell a hole from a row it has)")
            return False
        with self._lock:
            self._remaining = batch.remaining
        if batch.entries:
            info(f"Feed '{self.name}': the sweep brought {len(batch.entries)} entries from behind the cursor"
                 f"{f', {batch.remaining} still waiting' if batch.remaining else ''}")
        elif batch.remaining:
            info(f"Feed '{self.name}': the sweep stored nothing yet, {batch.remaining} waiting behind the cursor")
        else:
            trace(f"Feed '{self.name}': the sweep found nothing behind the cursor")
        self._store(FeedBatch(batch.entries, cursor=self._cursor))
        self._sweep_due = now if not batch.exhausted else now + self.SWEEP_SECONDS
        return not batch.exhausted

    def _trim_to_gap(self, batch: FeedBatch) -> tuple[FeedBatch, bool]:
        """While a gap is open, drop the entries the feed already covered (`_already_covered`);
        the rest, before the boundary or past the covered window, is stored. The gap is read
        once the batch's cursor reaches the resume cursor — an entry's timestamp says nothing
        about that for a cursor that is not time-ordered (an iMessage ROWID, where a send-later
        message dated weeks ahead sits among old rows) — or the feed runs dry before it does (a
        quiet gap): the cursor then carries on from where the batch got to, or jumps to the
        resume cursor, and the batch is not exhausted so the loop goes straight on. A batch
        that reached the resume cursor and dropped entries goes back to the resume cursor
        too: the rows past it were trimmed by timestamp as well, and a synced old message at
        a high ROWID would be lost for good. Those rows are read again with the gap closed;
        the history's dedup index absorbs what was stored twice. Returns (batch, gap closed)."""
        if self._gap_until is None:
            return batch, False
        inside = [e.timestamp for e in batch.entries if self._in_covered_window(e.timestamp)]
        stored = self._database().history_timestamps(self.name, inside) if inside else set()
        kept = [e for e in batch.entries if not (self._in_covered_window(e.timestamp) and e.timestamp in stored)]
        dropped = len(batch.entries) - len(kept)
        if len(inside) > dropped:
            info(f"Feed '{self.name}': {len(inside) - dropped} of {len(inside)} entries inside the covered window "
                 f"have no row in the history (a hole) and are kept")
        reached = self._feed.cursor_reached(batch.cursor, self._resume_cursor)
        if not reached and not batch.exhausted:
            if dropped:
                trace(f"Feed '{self.name}': {dropped} already-covered entries dropped, cursor {batch.cursor} "
                      f"not yet at {self._resume_cursor}")
            return FeedBatch(kept, cursor=batch.cursor, exhausted=False), False
        # A batch that dropped nothing lost nothing past the resume cursor, so it carries on
        # from where it got to; one that did is read again from the resume cursor.
        cursor = batch.cursor if reached and not dropped or self._resume_cursor is None else self._resume_cursor
        note(f"Feed '{self.name}': gap up to {time.strftime('%Y-%m-%d', time.localtime(self._gap_until))} read "
             f"({f'{dropped} already-covered entries dropped' if dropped else 'nothing already covered in the last batch'}"
             f"{'' if reached else ', the feed ran dry'}), carrying on at cursor={cursor}"
             f"{f' (the batch reached {batch.cursor}, read again with the gap closed)' if batch.cursor != cursor else ''}")
        return FeedBatch(kept, cursor=cursor, exhausted=False), True

    def _in_covered_window(self, timestamp: float) -> bool:
        """Inside the window the feed had stored before the move-back: from the boundary up
        to the newest entry it had, or unbounded when that edge was never recorded. Being
        inside is not being stored — a hole in the window (a batch the provider refused, rows
        a migration forgot) is told apart by the history's own rows in `_trim_to_gap`."""
        if timestamp < self._gap_until:
            return False
        return self._gap_covered_until is None or timestamp <= self._gap_covered_until

    def _store(self, batch: FeedBatch, gap_closed: bool = False):
        stored = 0
        if batch.entries:
            try:
                stored = self._history.add_context_messages(self.name, batch.entries)
            except Exception as e:
                self._unstored = (batch, gap_closed)
                info(f"Feed '{self.name}': {len(batch.entries)} entries not stored ({type(e).__name__}), "
                     f"cursor stays at {self._cursor}; the batch is held and stored again next pass")
                raise
            self._unstored = None
            newest = max(e.timestamp for e in batch.entries)
            with self._lock:
                self._stored += stored
                if self._state == FeedState.CATCHING_UP:
                    self._caught_up += stored
                if self._last_entry_ts is None or newest > self._last_entry_ts:
                    self._last_entry_ts = newest
            if self._state == FeedState.CATCHING_UP:
                fine(f"Feed '{self.name}': stored {stored} of {len(batch.entries)} entries "
                     f"(newest {time.strftime('%Y-%m-%d %H:%M', time.localtime(newest))})")
            elif stored:
                trace(f"Feed '{self.name}': {stored} new entr{'y' if stored == 1 else 'ies'}")
        with self._lock:
            cursor_moved = batch.cursor != self._cursor
            self._cursor = batch.cursor
            self._last_error = None
            if gap_closed:
                self._gap_until = self._resume_cursor = self._gap_covered_until = None
        if batch.entries or cursor_moved or gap_closed:
            # an empty live poll with an unchanged cursor has nothing to persist
            fields = dict(cursor=batch.cursor, last_entry_ts=self._last_entry_ts, live_since=self._live_since)
            if gap_closed:
                fields.update(gap_until=None, resume_cursor=None, gap_covered_until=None)
            self._database().save_feed(self.name, **fields)

    def _go_live(self):
        now = time.time()
        with self._lock:
            self._state = FeedState.LIVE
            self._live_since = now
            self._remaining = 0
        took = now - (self._live_boundary or now)
        caught_up = f"caught up {self._caught_up} row(s) in {took:.0f}s, " if self._caught_up else ""
        note(f"Feed '{self.name}' is live: {caught_up}{self._stored} rows in the history, "
             f"polling every {self._feed.poll_interval:.0f}s")
        self._database().save_feed(self.name, cursor=self._cursor, live_since=now)


def describe_feed_status(status: Optional[dict]) -> str:
    """The short phrase `list_integrations` and the dashboard show after the integration's
    name: "catching up 43%, 1203 left", "live", "stopped: no session file", "failed: …"."""
    if status is None:
        return "not started"
    state = status["state"]
    paused = "paused, " if status.get("paused") else ""
    if state == FeedState.CATCHING_UP.value:
        text = "catching up"
        if status["progress"] is not None:
            text += f" {status['progress'] * 100:.0f}%"
        if status["remaining"] is not None:
            text += f", {status['remaining']} left"
        return paused + text
    if state == FeedState.LIVE.value:
        sweeping = f", sweeping up {status['remaining']} missed" if status.get("remaining") else ""
        return paused + "live" + sweeping
    if state in (FeedState.STOPPED.value, FeedState.FAILED.value) and status["last_error"]:
        return f"{state}: {status['last_error']}"
    return state


class Feeds:
    """The running feeds by name. `start` replaces a runner already under that name (that is
    what `restart_integration` does through the bootstrap start hook)."""

    _instance: Optional['Feeds'] = None

    def __init__(self):
        self._runners: dict[str, FeedRunner] = {}
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> 'Feeds':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        if cls._instance is not None:
            cls._instance.stop_all()
        cls._instance = None

    @staticmethod
    def horizon_for(feed: Feed, now: Optional[float] = None) -> float:
        months = config.get(f"feeds.{feed.config_slug}.backfill_months", feed.default_backfill_months)
        try:
            months = float(months)
        except (TypeError, ValueError):
            info(f"Feed '{feed.feed_name}': feeds.{feed.config_slug}.backfill_months={months!r} is not "
                 f"a number, using the default {feed.default_backfill_months}")
            months = float(feed.default_backfill_months)
        return (now if now is not None else time.time()) - months * SECONDS_PER_MONTH

    def start(self, feed: Feed, message_history) -> Optional[FeedRunner]:
        if RUNNING_UNDER_TEST:
            info(f"Feed '{feed.feed_name}' not starting — running under test")
            return None
        self.stop(feed.feed_name)
        runner = FeedRunner(feed, message_history, self.horizon_for(feed))
        with self._lock:
            self._runners[feed.feed_name] = runner
        runner.start()
        info(f"Feed '{feed.feed_name}' started")
        return runner

    def stop(self, name: str):
        with self._lock:
            runner = self._runners.pop(name, None)
        if runner is None:
            return
        info(f"Stopping feed '{name}'...")
        runner.stop()

    def stop_all(self):
        with self._lock:
            names = list(self._runners)
        for name in names:
            self.stop(name)

    def runner(self, name: str) -> Optional[FeedRunner]:
        with self._lock:
            return self._runners.get(name)

    def names(self) -> list[str]:
        with self._lock:
            return list(self._runners)

    def find(self, name: str) -> Optional[FeedRunner]:
        """The runner whose feed name or config slug matches `name`, case-insensitively
        ("imessage" → iMessage, "photos" → iCloud Photos)."""
        wanted = name.strip().lower()
        with self._lock:
            for key, runner in self._runners.items():
                if key.lower() == wanted or runner._feed.config_slug == wanted:
                    return runner
        return None

    def status(self, name: str) -> Optional[dict]:
        runner = self.runner(name)
        return runner.status() if runner else None
