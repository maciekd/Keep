"""FeedRunner: catch up from the persisted cursor (or the horizon), go live, persist the cursor
after every stored batch, resume, give up after repeated failures (plans/FEED_PLAN.md §3.2)."""
import threading
import time
from unittest.mock import Mock, patch

import pytest

from assistant.integration.Feed import Feed, FeedBatch, FeedEntry, FEED_ROLE
from assistant.integration.FeedRunner import FeedRunner, FeedState, Feeds, SECONDS_PER_MONTH
from common.FloxyDatabase import FloxyDatabase


@pytest.fixture
def floxy_db(tmp_path):
    db = FloxyDatabase.__new__(FloxyDatabase)
    db.db_path = str(tmp_path / "floxy_test.db")
    db.lock = threading.Lock()
    db._init_database()
    return db


class RecordingHistory:
    """Records the batches without writing them: the runner's count grows by what it stores,
    the database keeps whatever `seed_rows` put there."""
    def __init__(self):
        self.batches: list[tuple[str, list[FeedEntry]]] = []
        self.notifies = 0

    def add_context_messages(self, source, entries):
        self.batches.append((source, list(entries)))
        self.notifies += 1
        return len(entries)


class DbHistory(RecordingHistory):
    """Writes the batches to the test database, the way MessageHistory does."""
    def __init__(self, db):
        super().__init__()
        self.db = db

    def add_context_messages(self, source, entries):
        super().add_context_messages(source, entries)
        return self.db.add_history_entries([(e.role, e.content, e.timestamp, e.location) for e in entries], source)


def seed_rows(db, count, source="Scripted", role=FEED_ROLE, start=0.0):
    """`count` history rows under `source`, what a feed stored before this run."""
    return db.add_history_entries([(role, f"{source} seeded #{i}", start + i + 0.5, None) for i in range(count)],
                                  source)


def seed_at(db, timestamps, source="Scripted", role=FEED_ROLE):
    """A history row of `source` at each of `timestamps`: what the feed had stored at those
    moments before a horizon move-back, so the gap trim can tell them from a hole."""
    return db.add_history_entries([(role, f"{source} stored @{ts}", ts, None) for ts in timestamps], source)


class ScriptedFeed(Feed):
    """Serves `pages` (each a list of timestamps) as successive batches; the cursor is the
    page index. Once the pages are gone every fetch is an empty live poll, and after
    `live_polls` of those the feed asks the runner to stop."""
    poll_interval = 0.0

    def __init__(self, pages, live_polls=1, count_remaining=True, name="Scripted"):
        self.pages = pages
        self.live_polls = live_polls
        self.count_remaining = count_remaining
        self._name = name
        self.calls: list[tuple] = []
        self.opened = 0
        self.closed = 0
        self.stop_event: threading.Event | None = None

    @property
    def feed_name(self):
        return self._name

    @property
    def config_slug(self):
        return "scripted"

    def initial_cursor(self, horizon_ts):
        return "0"

    def open_feed(self):
        self.opened += 1
        return True

    def close_feed(self):
        self.closed += 1

    def remaining(self, cursor):
        if not self.count_remaining:
            return None
        return sum(len(p) for p in self.pages[int(cursor or 0):])

    def fetch(self, cursor, backfill):
        index = int(cursor or 0)
        self.calls.append((index, backfill))
        if index < len(self.pages):
            entries = [FeedEntry(ts, f"msg@{ts}") for ts in self.pages[index]]
            return FeedBatch(entries, cursor=str(index + 1), exhausted=index + 1 >= len(self.pages))
        self.live_polls -= 1
        if self.live_polls <= 0 and self.stop_event is not None:
            self.stop_event.set()
        return FeedBatch([], cursor=cursor, exhausted=True)


class DoublePageFeed(ScriptedFeed):
    """The first fetch serves pages 0 and 1 together, cursor "2": the batch that reaches the
    resume cursor overshoots it, as a page of iMessage ROWIDs does."""
    def fetch(self, cursor, backfill):
        if self.calls:
            return super().fetch(cursor, backfill)
        self.calls.append((int(cursor or 0), backfill))
        entries = [FeedEntry(ts, f"msg@{ts}") for ts in self.pages[0] + self.pages[1]]
        return FeedBatch(entries, cursor="2", exhausted=False)


class SweepingFeed(ScriptedFeed):
    """Answers each sweep with the next of `sweeps` (a FeedBatch or None); once they are gone
    every sweep is empty and exhausted. Records (horizon_ts, cursor) per call."""
    def __init__(self, pages, sweeps, **kwargs):
        super().__init__(pages, **kwargs)
        self.sweeps = list(sweeps)
        self.sweep_calls: list[tuple] = []

    def sweep(self, horizon_ts, cursor):
        self.sweep_calls.append((horizon_ts, cursor))
        if self.sweeps:
            return self.sweeps.pop(0)
        return FeedBatch([], cursor=cursor, exhausted=True, remaining=0)


def run(runner: FeedRunner, feed: ScriptedFeed):
    feed.stop_event = runner._stop_event
    runner._run()
    return runner


NOW = time.time()
HORIZON = NOW - 3 * SECONDS_PER_MONTH


@pytest.mark.unit
class TestCatchUp:
    def test_first_run_starts_at_the_horizon_and_goes_live_after_the_last_page(self, floxy_db):
        feed = ScriptedFeed([[HORIZON + 10, HORIZON + 20], [HORIZON + 30]])
        history = RecordingHistory()
        runner = run(FeedRunner(feed, history, HORIZON, db=floxy_db), feed)

        # Two catch-up fetches (backfill=True), then a live poll (backfill=False).
        assert feed.calls == [(0, True), (1, True), (2, False)]
        assert [len(b[1]) for b in history.batches] == [2, 1]
        assert all(source == "Scripted" for source, _ in history.batches)
        assert history.notifies == 2
        assert runner.state == FeedState.STOPPED   # stopped by the feed's stop request after going live
        row = floxy_db.get_feed("Scripted")
        assert row["cursor"] == "2"
        assert runner.status()["stored"] == 3
        assert row["horizon"] == HORIZON
        assert row["live_since"] is not None
        assert row["last_entry_ts"] == HORIZON + 30

    def test_a_feed_reports_what_it_holds_to_the_telemetry_and_that_it_stopped(self, floxy_db):
        feed = ScriptedFeed([[HORIZON + 10, HORIZON + 20], [HORIZON + 30]])
        with patch("assistant.integration.FeedRunner.Telemetry") as telemetry:
            run(FeedRunner(feed, RecordingHistory(), HORIZON, db=floxy_db), feed)
        assert telemetry.get_instance().feed_count.call_args.args == ("Scripted", 3)
        telemetry.get_instance().feed_stopped.assert_called_once_with("Scripted")

    def test_cursor_is_persisted_after_every_batch(self, floxy_db):
        seen = []
        original = floxy_db.save_feed

        def spy(name, **fields):
            original(name, **fields)
            if "cursor" in fields:
                seen.append(floxy_db.get_feed(name)["cursor"])

        floxy_db.save_feed = spy
        feed = ScriptedFeed([[HORIZON + 1], [HORIZON + 2], [HORIZON + 3]])
        run(FeedRunner(feed, RecordingHistory(), HORIZON, db=floxy_db), feed)
        assert seen[:4] == ["0", "1", "2", "3"]

    def test_status_while_catching_up_counts_by_remaining(self, floxy_db):
        feed = ScriptedFeed([[HORIZON + 1, HORIZON + 2], [HORIZON + 3, HORIZON + 4]])
        runner = FeedRunner(feed, RecordingHistory(), HORIZON, db=floxy_db)
        statuses = []
        original_store = runner._store

        def store_and_snapshot(batch, *args):
            original_store(batch, *args)
            runner._refresh_remaining()
            statuses.append(runner.status())

        runner._store = store_and_snapshot
        run(runner, feed)
        first = statuses[0]
        assert first["state"] == "catching_up"
        assert (first["stored"], first["remaining"], first["total"]) == (2, 2, 4)
        assert first["progress"] == pytest.approx(0.5)
        assert statuses[-1]["state"] == "live"
        assert statuses[-1]["progress"] is None
        assert statuses[-1]["remaining"] == 0

    def test_progress_falls_back_to_timestamps_when_the_feed_cannot_count(self, floxy_db):
        midway = HORIZON + (NOW - HORIZON) / 2
        feed = ScriptedFeed([[midway], [NOW - 1]], count_remaining=False)
        runner = FeedRunner(feed, RecordingHistory(), HORIZON, db=floxy_db)
        progress = []
        original_store = runner._store

        def store_and_snapshot(batch, *args):
            original_store(batch, *args)
            progress.append(runner.status()["progress"])

        runner._store = store_and_snapshot
        run(runner, feed)
        assert progress[0] == pytest.approx(0.5, abs=0.01)
        assert runner.status()["remaining"] == 0

    def test_first_run_counts_the_rows_the_feed_already_has_in_the_history(self, floxy_db):
        """No feeds row (the table is younger than the feed's rows): the count is still the
        history's, and the catch-up's total is those rows plus what is ahead."""
        seed_rows(floxy_db, 5)
        seed_rows(floxy_db, 2, source="Other")
        feed = ScriptedFeed([[HORIZON + 1], [HORIZON + 2, HORIZON + 3]])
        runner = FeedRunner(feed, RecordingHistory(), HORIZON, db=floxy_db)
        runner._open()
        runner._load_cursor()
        status = runner.status()
        assert (status["stored"], status["remaining"], status["total"]) == (5, 3, 8)
        assert status["progress"] == pytest.approx(5 / 8)
        run(runner, feed)
        assert runner.status()["stored"] == 8

    def test_a_feed_with_its_own_store_is_counted_there_not_in_the_history(self, floxy_db):
        """Photos keeps its items in the index table, which predates its history rows: its
        `count_stored` answers and the history's count is never consulted."""
        seed_rows(floxy_db, 5)
        feed = ScriptedFeed([[HORIZON + 1]])
        feed.count_stored = lambda: 5000
        runner = FeedRunner(feed, RecordingHistory(), HORIZON, db=floxy_db)
        runner._open()
        runner._load_cursor()
        assert (runner.status()["stored"], runner.status()["total"]) == (5000, 5001)
        run(runner, feed)
        assert runner.status()["stored"] == 5001

    def test_the_count_is_recounted_from_the_history_as_the_loop_runs(self, floxy_db):
        """Rows purged behind the runner's back (hidden photos) come off the count without a restart."""
        seed_rows(floxy_db, 5)
        feed = ScriptedFeed([[HORIZON + 1]])
        runner = FeedRunner(feed, DbHistory(floxy_db), HORIZON, db=floxy_db)
        runner.RECOUNT_SECONDS = 0.0
        real_fetch = feed.fetch

        def purge_then_fetch(cursor, backfill):
            import sqlite3
            with sqlite3.connect(floxy_db.db_path) as conn:
                conn.execute("DELETE FROM history WHERE content IN ('Scripted seeded #0', 'Scripted seeded #1')")
            return real_fetch(cursor, backfill)

        feed.fetch = purge_then_fetch
        run(runner, feed)
        assert runner.status()["stored"] == 3 + 1
        assert floxy_db.count_history_entries("Scripted", FEED_ROLE) == 4

    def test_empty_batches_do_not_touch_the_history(self, floxy_db):
        feed = ScriptedFeed([[]], live_polls=2)
        history = RecordingHistory()
        run(FeedRunner(feed, history, HORIZON, db=floxy_db), feed)
        assert history.batches == []
        assert history.notifies == 0


@pytest.mark.unit
class TestResume:
    def test_second_run_resumes_from_the_stored_cursor(self, floxy_db):
        floxy_db.save_feed("Scripted", cursor="1", horizon=HORIZON, last_entry_ts=HORIZON + 5)
        seed_rows(floxy_db, 7)
        feed = ScriptedFeed([[HORIZON + 1], [HORIZON + 2]])
        history = RecordingHistory()
        runner = run(FeedRunner(feed, history, HORIZON, db=floxy_db), feed)
        assert feed.calls[0] == (1, True)
        assert [e.timestamp for _, batch in history.batches for e in batch] == [HORIZON + 2]
        assert runner.status()["stored"] == 8

    def test_restart_mid_catch_up_shows_the_same_total(self, floxy_db):
        """The total is the history's rows plus what is ahead of the cursor: a restart counts
        both afresh and lands on the same bar, with nothing to persist for it."""
        pages = [[HORIZON + 1], [HORIZON + 2], [HORIZON + 3], [HORIZON + 4]]
        feed = ScriptedFeed(pages)
        runner = FeedRunner(feed, RecordingHistory(), HORIZON, db=floxy_db)
        runner._load_cursor()
        assert (runner.status()["stored"], runner.status()["total"]) == (0, 4)
        assert runner.status()["progress"] == pytest.approx(0.0)

        floxy_db.save_feed("Scripted", cursor="1")     # one page stored, then a restart
        seed_rows(floxy_db, 1)
        feed = ScriptedFeed(pages)
        runner = FeedRunner(feed, RecordingHistory(), HORIZON, db=floxy_db)
        runner._load_cursor()
        status = runner.status()
        assert (status["stored"], status["remaining"], status["total"]) == (1, 3, 4)
        assert status["progress"] == pytest.approx(0.25)

        run(runner, feed)
        assert runner.status()["live_since"] is not None
        assert "total" not in floxy_db.get_feed("Scripted")

    def test_what_the_current_batch_has_finished_comes_off_remaining(self, floxy_db):
        """A Photos batch runs a hundred assets through its workers inside one fetch; the
        status follows them one by one through the feed's in_flight count."""
        feed = ScriptedFeed([[HORIZON + 1, HORIZON + 2, HORIZON + 3, HORIZON + 4]])
        feed.in_flight = lambda: 3
        runner = FeedRunner(feed, RecordingHistory(), HORIZON, db=floxy_db)
        runner._load_cursor()
        status = runner.status()
        # the three finished assets are done, not gone: they stay in the total
        assert (status["remaining"], status["total"]) == (1, 4)
        assert status["progress"] == pytest.approx(0.75)

        feed.in_flight = lambda: 9      # never below zero
        assert runner.status()["remaining"] == 0

        run(runner, feed)
        assert runner.status()["remaining"] == 0, "live: nothing is in flight any more"

    def test_eta_comes_from_this_processs_own_rate_over_its_unpaused_time(self, floxy_db):
        """8 items left at start; 60 s later 6 are done (4 stored, 2 in flight): 2 left at
        6 per minute = 20 s. A pause does not count as time spent; before anything is
        measured there is no ETA at all."""
        pages = [[HORIZON + 1, HORIZON + 2], [HORIZON + 3, HORIZON + 4],
                 [HORIZON + 5, HORIZON + 6], [HORIZON + 7, HORIZON + 8]]
        feed = ScriptedFeed(pages)
        runner = FeedRunner(feed, RecordingHistory(), HORIZON, db=floxy_db)
        t0 = 1_000_000.0
        with patch("assistant.integration.FeedRunner.time.time", return_value=t0):
            runner._load_cursor()
            for _ in range(2):
                runner._store(feed.fetch(runner._cursor, backfill=True))
                runner._refresh_remaining()
            assert runner.status()["eta_seconds"] is None, "nothing measured yet"
        feed.in_flight = lambda: 2
        with patch("assistant.integration.FeedRunner.time.time", return_value=t0 + 60):
            status = runner.status()
            assert status["remaining"] == 2
            assert status["eta_seconds"] == pytest.approx(20.0)
            runner.pause()
        with patch("assistant.integration.FeedRunner.time.time", return_value=t0 + 600):
            assert runner.status()["eta_seconds"] == pytest.approx(20.0), "the clock stops while paused"
            runner.resume()
        with patch("assistant.integration.FeedRunner.time.time", return_value=t0 + 660):
            assert runner.status()["eta_seconds"] == pytest.approx(40.0), "6 done in 120 unpaused seconds"

    def test_eta_needs_a_minimum_of_items_and_time(self, floxy_db):
        feed = ScriptedFeed([[HORIZON + i for i in range(1, 11)]])
        runner = FeedRunner(feed, RecordingHistory(), HORIZON, db=floxy_db)
        t0 = 1_000_000.0
        with patch("assistant.integration.FeedRunner.time.time", return_value=t0):
            runner._load_cursor()
        feed.in_flight = lambda: FeedRunner.ETA_MIN_ITEMS - 1
        with patch("assistant.integration.FeedRunner.time.time", return_value=t0 + 3600):
            assert runner.status()["eta_seconds"] is None, "too few items"
        feed.in_flight = lambda: FeedRunner.ETA_MIN_ITEMS
        with patch("assistant.integration.FeedRunner.time.time", return_value=t0 + FeedRunner.ETA_MIN_SECONDS - 1):
            assert runner.status()["eta_seconds"] is None, "too little time"
        with patch("assistant.integration.FeedRunner.time.time", return_value=t0 + FeedRunner.ETA_MIN_SECONDS):
            assert runner.status()["eta_seconds"] == pytest.approx(5 * FeedRunner.ETA_MIN_SECONDS / 5)

    def test_a_feed_whose_in_flight_raises_is_counted_as_zero(self, floxy_db):
        feed = ScriptedFeed([[HORIZON + 1, HORIZON + 2]])
        feed.in_flight = Mock(side_effect=RuntimeError("boom"))
        runner = FeedRunner(feed, RecordingHistory(), HORIZON, db=floxy_db)
        runner._load_cursor()
        assert runner.status()["remaining"] == 2

    def test_horizon_moved_further_back_restarts_from_the_new_horizon(self, floxy_db):
        floxy_db.save_feed("Scripted", cursor="2", horizon=HORIZON)
        feed = ScriptedFeed([[HORIZON + 1], [HORIZON + 2]])
        older = HORIZON - 2 * SECONDS_PER_MONTH
        runner = run(FeedRunner(feed, RecordingHistory(), older, db=floxy_db), feed)
        assert feed.calls[0] == (0, True)
        assert floxy_db.get_feed("Scripted")["horizon"] == older
        assert runner.status()["horizon_ts"] == older

    def test_horizon_moved_back_reads_only_the_gap_then_resumes_at_the_old_cursor(self, floxy_db):
        # Pages 0-1 are the gap; page 1 straddles the old horizon; pages 2-3 are what the
        # feed had left when the horizon moved (its cursor was "2").
        floxy_db.save_feed("Scripted", cursor="2", horizon=HORIZON, last_entry_ts=HORIZON + 40)
        seed_rows(floxy_db, 3)
        seed_at(floxy_db, [HORIZON + 5])    # the straddling entry the feed had stored before the move
        older = HORIZON - 2 * SECONDS_PER_MONTH
        feed = ScriptedFeed([[older + 1], [older + 2, HORIZON + 5], [HORIZON + 50], [HORIZON + 60]])
        history = RecordingHistory()
        runner = run(FeedRunner(feed, history, older, db=floxy_db), feed)

        assert feed.calls == [(0, True), (1, True), (2, True), (3, True), (4, False)]
        assert [[e.timestamp for e in b] for _, b in history.batches] == \
            [[older + 1], [older + 2], [HORIZON + 50], [HORIZON + 60]]
        row = floxy_db.get_feed("Scripted")
        assert (row["cursor"], row["gap_until"], row["resume_cursor"]) == ("4", None, None)
        assert runner.status()["stored"] == 4 + 4
        assert runner.state == FeedState.STOPPED

    def test_a_quiet_gap_that_runs_dry_still_jumps_to_the_resume_cursor(self, floxy_db):
        floxy_db.save_feed("Scripted", cursor="1", horizon=HORIZON)
        older = HORIZON - 2 * SECONDS_PER_MONTH
        feed = ScriptedFeed([[older + 1]], live_polls=2)
        history = RecordingHistory()
        run(FeedRunner(feed, history, older, db=floxy_db), feed)
        # the only page is exhausted without reaching the boundary: jump, one more backfill
        # fetch from the resume cursor, then live
        assert feed.calls == [(0, True), (1, True), (1, False)]
        assert [[e.timestamp for e in b] for _, b in history.batches] == [[older + 1]]
        assert floxy_db.get_feed("Scripted")["gap_until"] is None

    def test_gap_state_is_persisted_so_a_restart_carries_on_mid_gap(self, floxy_db):
        older = HORIZON - 2 * SECONDS_PER_MONTH
        floxy_db.save_feed("Scripted", cursor="1", horizon=older, gap_until=HORIZON, resume_cursor="3")
        seed_at(floxy_db, [HORIZON + 5])
        feed = ScriptedFeed([[older + 1], [older + 2], [HORIZON + 5], [HORIZON + 50]])
        history = RecordingHistory()
        run(FeedRunner(feed, history, older, db=floxy_db), feed)
        assert feed.calls == [(1, True), (2, True), (3, True), (4, False)]
        assert [[e.timestamp for e in b] for _, b in history.batches] == [[older + 2], [HORIZON + 50]]

    def test_gap_fields_survive_a_history_write_failure(self, floxy_db):
        older = HORIZON - 2 * SECONDS_PER_MONTH
        floxy_db.save_feed("Scripted", cursor="0", horizon=older, gap_until=HORIZON, resume_cursor="1")
        seed_at(floxy_db, [HORIZON + 5])
        feed = ScriptedFeed([[older + 1, HORIZON + 5], [HORIZON + 50]])
        history = RecordingHistory()
        original = history.add_context_messages
        failures = iter([True, False, False, False])

        def flaky(source, entries):
            if next(failures, False):
                raise RuntimeError("disk full")
            return original(source, entries)

        history.add_context_messages = flaky
        run(FeedRunner(feed, history, older, db=floxy_db), feed)
        # the trimmed boundary batch is held and stored on the next pass, not refetched
        assert feed.calls[:2] == [(0, True), (1, True)]
        assert [[e.timestamp for e in b] for _, b in history.batches] == [[older + 1], [HORIZON + 50]]
        assert floxy_db.get_feed("Scripted")["gap_until"] is None

    def test_a_gap_keeps_entries_past_the_covered_window(self, floxy_db):
        # The feed had stored up to HORIZON+40 when the horizon moved back. The gap batch spills
        # past the boundary into that window (dropped: the history has the row) and beyond it
        # into entries never stored (kept: Photos describes those with the LLM and lists them in
        # its ledger, a drop loses them).
        floxy_db.save_feed("Scripted", cursor="1", horizon=HORIZON, last_entry_ts=HORIZON + 40)
        seed_at(floxy_db, [HORIZON + 5])
        older = HORIZON - 2 * SECONDS_PER_MONTH
        feed = ScriptedFeed([[older + 1, HORIZON + 5, HORIZON + 45], [HORIZON + 60]])
        history = RecordingHistory()
        run(FeedRunner(feed, history, older, db=floxy_db), feed)
        assert [[e.timestamp for e in b] for _, b in history.batches] == [[older + 1, HORIZON + 45], [HORIZON + 60]]
        row = floxy_db.get_feed("Scripted")
        assert (row["gap_until"], row["resume_cursor"], row["gap_covered_until"]) == (None, None, None)

    def test_the_covered_edge_is_recorded_when_the_gap_opens(self, floxy_db):
        floxy_db.save_feed("Scripted", cursor="2", horizon=HORIZON, last_entry_ts=HORIZON + 40)
        older = HORIZON - 2 * SECONDS_PER_MONTH
        runner = FeedRunner(ScriptedFeed([]), RecordingHistory(), older, db=floxy_db)
        runner._open()
        runner._load_cursor()
        row = floxy_db.get_feed("Scripted")
        assert (row["gap_until"], row["resume_cursor"], row["gap_covered_until"]) == (HORIZON, "2", HORIZON + 40)

    def test_a_feed_that_had_stored_nothing_covers_nothing_past_the_boundary(self, floxy_db):
        floxy_db.save_feed("Scripted", cursor="0", horizon=HORIZON)
        older = HORIZON - 2 * SECONDS_PER_MONTH
        feed = ScriptedFeed([[older + 1, HORIZON + 5]])
        history = RecordingHistory()
        run(FeedRunner(feed, history, older, db=floxy_db), feed)
        assert [[e.timestamp for e in b] for _, b in history.batches] == [[older + 1, HORIZON + 5]]

    def test_the_gap_closes_when_the_cursor_reaches_the_resume_cursor_not_on_a_stray_timestamp(self, floxy_db):
        # iMessage reads by ROWID: a send-later message dated weeks ahead sits among old rows.
        # Page 0 carries one, but the cursor is still short of the resume cursor, so the gap
        # stays open and page 1 is read instead of skipped.
        floxy_db.save_feed("Scripted", cursor="2", horizon=HORIZON, last_entry_ts=HORIZON + 40)
        seed_at(floxy_db, [HORIZON + 20])
        older = HORIZON - 2 * SECONDS_PER_MONTH
        feed = ScriptedFeed([[older + 1, HORIZON + 20], [older + 2], [HORIZON + 50]])
        history = RecordingHistory()
        run(FeedRunner(feed, history, older, db=floxy_db), feed)
        assert feed.calls == [(0, True), (1, True), (2, True), (3, False)]
        assert [[e.timestamp for e in b] for _, b in history.batches] == [[older + 1], [older + 2], [HORIZON + 50]]

    def test_a_batch_that_overshoots_the_resume_cursor_is_read_again_with_the_gap_closed(self, floxy_db):
        # The feed's cursor was "1" when the horizon moved back. Page 1 is past it — rows the
        # feed never stored — but one of them is dated inside the covered window (an old
        # message synced in at a high ROWID, whose live row the history holds). The first fetch
        # spills into page 1, so the trim drops that row; carrying on from the batch's cursor "2"
        # would lose it for good. The cursor goes back to "1" instead and page 1 is read again
        # with the gap closed; the entry past the window is stored twice, which the history's
        # dedup index absorbs.
        floxy_db.save_feed("Scripted", cursor="1", horizon=HORIZON, last_entry_ts=HORIZON + 40)
        seed_at(floxy_db, [HORIZON + 5])
        older = HORIZON - 2 * SECONDS_PER_MONTH
        feed = DoublePageFeed([[older + 1], [HORIZON + 5, HORIZON + 45], [HORIZON + 60]])
        history = DbHistory(floxy_db)
        runner = run(FeedRunner(feed, history, older, db=floxy_db), feed)
        assert feed.calls == [(0, True), (1, True), (2, True), (3, False)]
        assert [[e.timestamp for e in b] for _, b in history.batches] == \
            [[older + 1, HORIZON + 45], [HORIZON + 5, HORIZON + 45], [HORIZON + 60]]
        # the seeded live row plus the four backfill rows (the re-read HORIZON+5 differs in content)
        assert floxy_db.count_history_entries("Scripted", FEED_ROLE) == 5
        assert runner.status()["stored"] == 5
        row = floxy_db.get_feed("Scripted")
        assert (row["cursor"], row["gap_until"], row["resume_cursor"]) == ("3", None, None)

    def test_moved_back_again_mid_gap_keeps_the_boundary_and_resume_cursor(self, floxy_db):
        one_back = HORIZON - SECONDS_PER_MONTH
        two_back = HORIZON - 2 * SECONDS_PER_MONTH
        floxy_db.save_feed("Scripted", cursor="1", horizon=one_back, gap_until=HORIZON, resume_cursor="3")
        seed_at(floxy_db, [HORIZON + 5])
        feed = ScriptedFeed([[two_back + 1], [one_back + 1], [HORIZON + 5], [HORIZON + 50]])
        history = RecordingHistory()
        run(FeedRunner(feed, history, two_back, db=floxy_db), feed)
        assert feed.calls == [(0, True), (1, True), (2, True), (3, True), (4, False)]
        assert [[e.timestamp for e in b] for _, b in history.batches] == \
            [[two_back + 1], [one_back + 1], [HORIZON + 50]]
        row = floxy_db.get_feed("Scripted")
        assert (row["horizon"], row["gap_until"], row["resume_cursor"]) == (two_back, None, None)

    def test_remaining_during_a_gap_counts_only_the_gap(self, floxy_db):
        floxy_db.save_feed("Scripted", cursor="2", horizon=HORIZON)
        seed_rows(floxy_db, 3)
        older = HORIZON - 2 * SECONDS_PER_MONTH
        feed = ScriptedFeed([[older + 1, older + 2], [older + 3], [HORIZON + 50, HORIZON + 51, HORIZON + 52]])
        runner = FeedRunner(feed, RecordingHistory(), older, db=floxy_db)
        runner._open()
        runner._load_cursor()
        # 6 items ahead of cursor "0", 3 of them past the resume cursor "2"
        status = runner.status()
        assert (status["stored"], status["remaining"], status["total"]) == (3, 3, 6)

    def test_horizon_moved_forward_keeps_the_stored_one(self, floxy_db):
        floxy_db.save_feed("Scripted", cursor="2", horizon=HORIZON)
        feed = ScriptedFeed([[HORIZON + 1], [HORIZON + 2]])
        runner = run(FeedRunner(feed, RecordingHistory(), HORIZON + SECONDS_PER_MONTH, db=floxy_db), feed)
        assert feed.calls[0] == (2, False) or feed.calls[0] == (2, True)
        assert runner.status()["horizon_ts"] == HORIZON
        assert floxy_db.get_feed("Scripted")["horizon"] == HORIZON

    def test_a_hole_inside_the_covered_window_is_kept_while_a_stored_sibling_is_dropped(self, floxy_db):
        # Both entries sit inside the window the feed had covered (HORIZON .. HORIZON+40), but
        # only one has a row: the other was lost to a batch the provider refused, and the
        # move-back is its one chance to be stored. Being inside the window is no proof of a row.
        floxy_db.save_feed("Scripted", cursor="1", horizon=HORIZON, last_entry_ts=HORIZON + 40)
        seed_at(floxy_db, [HORIZON + 10])
        older = HORIZON - 2 * SECONDS_PER_MONTH
        feed = ScriptedFeed([[older + 1, HORIZON + 10, HORIZON + 20], [HORIZON + 60]])
        history = RecordingHistory()
        run(FeedRunner(feed, history, older, db=floxy_db), feed)
        assert [[e.timestamp for e in b] for _, b in history.batches] == [[older + 1, HORIZON + 20], [HORIZON + 60]]

    def test_a_row_of_another_feed_at_the_same_moment_does_not_cover_an_entry(self, floxy_db):
        floxy_db.save_feed("Scripted", cursor="1", horizon=HORIZON, last_entry_ts=HORIZON + 40)
        seed_at(floxy_db, [HORIZON + 10], source="Other")
        older = HORIZON - 2 * SECONDS_PER_MONTH
        feed = ScriptedFeed([[older + 1, HORIZON + 10], [HORIZON + 60]])
        history = RecordingHistory()
        run(FeedRunner(feed, history, older, db=floxy_db), feed)
        assert [[e.timestamp for e in b] for _, b in history.batches] == [[older + 1, HORIZON + 10], [HORIZON + 60]]


@pytest.mark.unit
class TestSweep:
    """Once live and idle the runner asks the feed to sweep for items behind the cursor it has
    no row for; what comes back is stored without moving the cursor."""

    def test_a_live_idle_feed_is_swept_and_the_cursor_stays(self, floxy_db):
        swept = FeedBatch([FeedEntry(HORIZON - 50, "swept")], cursor="999", exhausted=True, remaining=7)
        feed = SweepingFeed([[HORIZON + 1]], [swept])
        history = RecordingHistory()
        runner = run(FeedRunner(feed, history, HORIZON, db=floxy_db), feed)
        assert [[e.timestamp for e in b] for _, b in history.batches] == [[HORIZON + 1], [HORIZON - 50]]
        assert feed.sweep_calls == [(HORIZON, "1")]
        assert floxy_db.get_feed("Scripted")["cursor"] == "1", "a sweep batch's cursor is ignored"
        status = runner.status()
        assert (status["remaining"], status["stored"]) == (7, 2)
        assert status["live_since"] is not None

    def test_an_unexhausted_sweep_is_followed_at_once_and_an_exhausted_one_waits(self, floxy_db):
        more = FeedBatch([FeedEntry(HORIZON - 50, "a")], exhausted=False, remaining=1)
        done = FeedBatch([FeedEntry(HORIZON - 40, "b")], exhausted=True, remaining=0)
        feed = SweepingFeed([[HORIZON + 1]], [more, done], live_polls=4)
        history = RecordingHistory()
        runner = run(FeedRunner(feed, history, HORIZON, db=floxy_db), feed)
        # four idle polls, but the sweep after `done` is not due for SWEEP_SECONDS
        assert len(feed.sweep_calls) == 2
        assert [[e.timestamp for e in b] for _, b in history.batches] == [[HORIZON + 1], [HORIZON - 50], [HORIZON - 40]]
        assert runner._sweep_due > time.time() + FeedRunner.SWEEP_SECONDS - 60

    def test_a_feed_that_cannot_sweep_is_asked_once(self, floxy_db):
        feed = SweepingFeed([[HORIZON + 1]], [None], live_polls=3)
        runner = run(FeedRunner(feed, RecordingHistory(), HORIZON, db=floxy_db), feed)
        assert len(feed.sweep_calls) == 1
        assert runner._sweep_due == float("inf")

    def test_no_sweep_while_catching_up(self, floxy_db):
        feed = SweepingFeed([[HORIZON + 1], [HORIZON + 2], [HORIZON + 3]], [])
        run(FeedRunner(feed, RecordingHistory(), HORIZON, db=floxy_db), feed)
        # three catch-up fetches, one live poll: the one sweep came after the catch-up, at its final cursor
        assert feed.calls == [(0, True), (1, True), (2, True), (3, False)]
        assert feed.sweep_calls == [(HORIZON, "3")]

    def test_no_sweep_mid_gap(self, floxy_db):
        floxy_db.save_feed("Scripted", cursor="2", horizon=HORIZON, last_entry_ts=HORIZON + 40)
        older = HORIZON - 2 * SECONDS_PER_MONTH
        feed = SweepingFeed([[older + 1], [older + 2], [HORIZON + 50]], [])
        run(FeedRunner(feed, RecordingHistory(), older, db=floxy_db), feed)
        assert feed.sweep_calls == [(older, "3")], "swept once, after the gap closed and the feed went live"

    def test_a_sweep_batch_the_history_refuses_is_held_and_stored_next_pass(self, floxy_db):
        swept = FeedBatch([FeedEntry(HORIZON - 50, "swept")], exhausted=True, remaining=0)
        feed = SweepingFeed([[HORIZON + 1]], [swept], live_polls=3)
        history = RecordingHistory()
        real_add = history.add_context_messages
        failed = []

        def flaky(source, entries):
            if entries and entries[0].content == "swept" and not failed:
                failed.append(True)
                raise RuntimeError("database is locked")
            return real_add(source, entries)

        history.add_context_messages = flaky
        runner = run(FeedRunner(feed, history, HORIZON, db=floxy_db), feed)
        assert [[e.timestamp for e in b] for _, b in history.batches] == [[HORIZON + 1], [HORIZON - 50]]
        assert feed.sweeps == [], "the held batch was stored, not swept for again"
        assert floxy_db.get_feed("Scripted")["cursor"] == "1"
        assert runner.status()["stored"] == 2

    def test_a_sweep_that_raises_counts_as_a_failure_and_the_feed_carries_on(self, floxy_db):
        feed = SweepingFeed([[HORIZON + 1]], [], live_polls=3)
        feed.sweep = Mock(side_effect=[RuntimeError("library locked"), FeedBatch([], exhausted=True, remaining=0)])
        runner = run(FeedRunner(feed, RecordingHistory(), HORIZON, db=floxy_db), feed)
        assert runner.state != FeedState.FAILED
        assert runner._failures == 0, "the next clean pass clears the failure"
        assert feed.sweep.call_count == 2, "a sweep that raised is due again at the next idle poll"


@pytest.mark.unit
class TestDescribeFeedStatus:
    def _status(self, **fields):
        base = {"state": FeedState.LIVE.value, "paused": False, "remaining": None, "progress": None,
                "last_error": None}
        return {**base, **fields}

    def test_a_live_feed_sweeping_shows_how_many_are_left(self):
        from assistant.integration.FeedRunner import describe_feed_status
        text = describe_feed_status(self._status(remaining=2273))
        assert text.startswith("live") and "2273" in text

    def test_a_live_feed_with_nothing_behind_the_cursor_is_just_live(self):
        from assistant.integration.FeedRunner import describe_feed_status
        assert describe_feed_status(self._status(remaining=0)) == describe_feed_status(self._status(remaining=None))
        assert not any(ch.isdigit() for ch in describe_feed_status(self._status(remaining=0)))


@pytest.mark.unit
class TestFailures:
    def test_open_returning_false_stops_with_the_reason(self, floxy_db):
        feed = ScriptedFeed([[HORIZON + 1]])
        feed.open_feed = lambda: False
        type(feed).open_error = property(lambda self: "no session file")
        try:
            runner = run(FeedRunner(feed, RecordingHistory(), HORIZON, db=floxy_db), feed)
        finally:
            del type(feed).open_error
        assert runner.state == FeedState.STOPPED
        assert runner.status()["last_error"] == "no session file"
        assert feed.calls == []
        assert floxy_db.get_feed("Scripted") is None

    def test_open_raising_marks_the_feed_failed(self, floxy_db):
        feed = ScriptedFeed([[HORIZON + 1]])
        feed.open_feed = Mock(side_effect=RuntimeError("chat.db locked"))
        runner = run(FeedRunner(feed, RecordingHistory(), HORIZON, db=floxy_db), feed)
        assert runner.state == FeedState.FAILED
        assert "chat.db locked" in runner.status()["last_error"]

    def test_repeated_fetch_failures_give_up_and_keep_the_cursor(self, floxy_db):
        feed = ScriptedFeed([[HORIZON + 1], [HORIZON + 2]])
        real_fetch = feed.fetch

        def flaky(cursor, backfill):
            if cursor == "1":
                raise ConnectionError("gone")
            return real_fetch(cursor, backfill)

        feed.fetch = flaky
        runner = FeedRunner(feed, RecordingHistory(), HORIZON, db=floxy_db)
        runner.FAILURE_LIMIT = 3
        run(runner, feed)
        assert runner.state == FeedState.FAILED
        assert "gone" in runner.status()["last_error"]
        assert floxy_db.get_feed("Scripted")["cursor"] == "1"
        assert feed.closed == 1

    def test_a_single_failure_is_retried(self, floxy_db):
        feed = ScriptedFeed([[HORIZON + 1]])
        real_fetch = feed.fetch
        failed = []

        def once(cursor, backfill):
            if not failed:
                failed.append(True)
                raise ConnectionError("blip")
            return real_fetch(cursor, backfill)

        feed.fetch = once
        runner = run(FeedRunner(feed, RecordingHistory(), HORIZON, db=floxy_db), feed)
        assert runner.status()["last_error"] is None
        assert floxy_db.get_feed("Scripted")["cursor"] == "1"

    def test_history_write_failure_does_not_advance_the_cursor(self, floxy_db):
        feed = ScriptedFeed([[HORIZON + 1]])
        history = Mock()
        history.add_context_messages.side_effect = RuntimeError("disk full")
        runner = FeedRunner(feed, history, HORIZON, db=floxy_db)
        runner.FAILURE_LIMIT = 2
        run(runner, feed)
        assert runner.state == FeedState.FAILED
        assert floxy_db.get_feed("Scripted")["cursor"] == "0"

    def test_history_write_failure_holds_the_batch_and_stores_it_next_pass_without_refetching(self, floxy_db):
        feed = ScriptedFeed([[HORIZON + 1, HORIZON + 2]])
        history = RecordingHistory()
        real_add = history.add_context_messages
        failed = []

        def flaky(source, entries):
            if not failed:
                failed.append(True)
                raise RuntimeError("database is locked")
            return real_add(source, entries)

        history.add_context_messages = flaky
        saved = []
        original = floxy_db.save_feed
        floxy_db.save_feed = lambda name, **f: (saved.append(dict(f)), original(name, **f))
        runner = run(FeedRunner(feed, history, HORIZON, db=floxy_db), feed)

        # The failed batch was not persisted (only the first-run row, then the stored one); the
        # page was fetched once — its entries can be LLM output the feed will not produce again.
        assert feed.calls[:2] == [(0, True), (1, False)]
        assert [len(b[1]) for b in history.batches] == [2]
        assert [f["cursor"] for f in saved if "cursor" in f][:2] == ["0", "1"]

    def test_a_held_batch_gets_one_last_store_when_the_runner_gives_up(self, floxy_db):
        feed = ScriptedFeed([[HORIZON + 1]])
        history = RecordingHistory()
        real_add = history.add_context_messages
        attempts = []

        def flaky(source, entries):
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("database is locked")
            return real_add(source, entries)

        history.add_context_messages = flaky
        runner = FeedRunner(feed, history, HORIZON, db=floxy_db)
        runner.FAILURE_LIMIT = 2
        run(runner, feed)
        assert runner.state == FeedState.FAILED
        assert [len(b[1]) for b in history.batches] == [1]
        assert floxy_db.get_feed("Scripted")["cursor"] == "1"
        assert runner.status()["stored"] == 1

    def test_the_stored_batch_clears_the_error_and_counts(self, floxy_db):
        feed = ScriptedFeed([[HORIZON + 1, HORIZON + 2]])
        history = RecordingHistory()
        real_add = history.add_context_messages
        failed = []

        def flaky(source, entries):
            if not failed:
                failed.append(True)
                raise RuntimeError("database is locked")
            return real_add(source, entries)

        history.add_context_messages = flaky
        runner = run(FeedRunner(feed, history, HORIZON, db=floxy_db), feed)
        assert runner.status()["last_error"] is None
        assert floxy_db.get_feed("Scripted")["cursor"] == "1"
        assert runner.status()["stored"] == 2


@pytest.mark.unit
class TestCursorReached:
    """The default `Feed.cursor_reached`: every feed's cursor is a number as text."""
    def test_numeric_cursors_compare_as_numbers(self):
        feed = ScriptedFeed([])
        assert feed.cursor_reached("1500", "1500")
        assert feed.cursor_reached("1501", "1500")
        assert not feed.cursor_reached("999", "1500")
        assert feed.cursor_reached("2.0", "1.5")
        assert not feed.cursor_reached("1.5", "2.0")

    def test_no_target_is_reached_at_once_and_no_cursor_never_is(self):
        feed = ScriptedFeed([])
        assert feed.cursor_reached(None, None)
        assert feed.cursor_reached("5", None)
        assert not feed.cursor_reached(None, "5")

    def test_cursors_that_are_not_numbers_fall_back_to_equality(self):
        feed = ScriptedFeed([])
        assert feed.cursor_reached("abc", "abc")
        assert not feed.cursor_reached("abd", "abc")


@pytest.mark.unit
class TestThread:
    def test_start_and_stop_close_the_feed(self, floxy_db):
        feed = ScriptedFeed([[HORIZON + 1]], live_polls=10 ** 6)
        feed.poll_interval = 0.01
        runner = FeedRunner(feed, RecordingHistory(), HORIZON, db=floxy_db)
        runner.start()
        deadline = time.time() + 5
        while runner.state != FeedState.LIVE and time.time() < deadline:
            time.sleep(0.01)
        assert runner.state == FeedState.LIVE
        runner.stop()
        assert runner.state == FeedState.STOPPED
        assert feed.opened == 1
        assert feed.closed >= 1
        assert not runner._thread.is_alive()


@pytest.mark.unit
class TestWaitOpened:
    """A restart's health check waits for open_feed to have come back, either way."""

    def test_set_once_the_feed_is_open(self, floxy_db):
        feed = ScriptedFeed([[HORIZON + 1]], live_polls=10 ** 6)
        feed.poll_interval = 0.01
        runner = FeedRunner(feed, RecordingHistory(), HORIZON, db=floxy_db)
        assert runner.wait_opened(0.01) is False
        runner.start()
        try:
            assert runner.wait_opened(5) is True
            assert feed.opened == 1
        finally:
            runner.stop()

    def test_set_when_open_refuses_or_raises(self, floxy_db):
        refusing = ScriptedFeed([])
        refusing.open_feed = lambda: False
        runner = FeedRunner(refusing, RecordingHistory(), HORIZON, db=floxy_db)
        runner.start()
        assert runner.wait_opened(5) is True
        assert runner.state == FeedState.STOPPED

        def boom():
            raise RuntimeError("no network")
        raising = ScriptedFeed([])
        raising.open_feed = boom
        runner = FeedRunner(raising, RecordingHistory(), HORIZON, db=floxy_db)
        runner.start()
        assert runner.wait_opened(5) is True
        assert runner.state == FeedState.FAILED


@pytest.mark.unit
class TestPause:
    def _wait(self, predicate, timeout=5.0):
        deadline = time.time() + timeout
        while not predicate() and time.time() < deadline:
            time.sleep(0.01)
        return predicate()

    def test_pause_holds_fetching_and_resume_carries_on(self, floxy_db):
        feed = ScriptedFeed([[HORIZON + 1]], live_polls=10 ** 6)
        feed.poll_interval = 0.01
        runner = FeedRunner(feed, RecordingHistory(), HORIZON, db=floxy_db)
        runner.start()
        assert self._wait(lambda: runner.state == FeedState.LIVE)
        parked = threading.Event()
        wait_for_resume = runner._running.wait

        def park():
            parked.set()
            wait_for_resume()

        runner._running.wait = park
        assert runner.pause() and runner.paused
        assert not runner.pause()   # already paused
        assert parked.wait(5)       # the loop is idling on the resume, not fetching
        calls_when_paused = len(feed.calls)
        assert runner.state == FeedState.LIVE
        assert runner.status()["paused"] is True
        assert runner.resume() and not runner.paused
        assert not runner.resume()  # not paused
        assert self._wait(lambda: len(feed.calls) > calls_when_paused)
        runner.stop()

    def test_stop_wakes_a_paused_runner(self, floxy_db):
        feed = ScriptedFeed([[HORIZON + 1]], live_polls=10 ** 6)
        feed.poll_interval = 0.01
        runner = FeedRunner(feed, RecordingHistory(), HORIZON, db=floxy_db)
        runner.start()
        assert self._wait(lambda: runner.state == FeedState.LIVE)
        runner.pause()
        runner.stop(timeout=2.0)
        assert not runner._thread.is_alive()
        assert runner.state == FeedState.STOPPED

@pytest.mark.unit
class TestFeedsRegistry:
    @pytest.fixture(autouse=True)
    def fresh(self):
        Feeds.reset_instance()
        yield
        Feeds.reset_instance()

    def test_horizon_comes_from_config_or_the_feed_default(self):
        feed = ScriptedFeed([])
        with patch("assistant.integration.FeedRunner.config") as cfg:
            cfg.get.side_effect = lambda key, default=None: {"feeds.scripted.backfill_months": 6}.get(key, default)
            assert Feeds.horizon_for(feed, now=1000.0) == pytest.approx(1000.0 - 6 * SECONDS_PER_MONTH)
            cfg.get.side_effect = lambda key, default=None: default
            assert Feeds.horizon_for(feed, now=1000.0) == pytest.approx(1000.0 - SECONDS_PER_MONTH)
            cfg.get.side_effect = lambda key, default=None: "lots"
            assert Feeds.horizon_for(feed, now=1000.0) == pytest.approx(1000.0 - SECONDS_PER_MONTH)

    def test_start_replaces_a_runner_of_the_same_name(self, floxy_db):
        feeds = Feeds.get_instance()
        started, stopped = [], []
        with patch("assistant.integration.FeedRunner.RUNNING_UNDER_TEST", False), \
                patch("assistant.integration.FeedRunner.FeedRunner") as runner_cls:
            runner_cls.side_effect = lambda *a, **k: Mock(start=lambda: started.append(a[0]),
                                                          stop=lambda timeout=10.0: stopped.append(a[0]))
            first = ScriptedFeed([], name="Same")
            second = ScriptedFeed([], name="Same")
            with patch.object(Feeds, "horizon_for", return_value=HORIZON):
                feeds.start(first, RecordingHistory())
                feeds.start(second, RecordingHistory())
        assert started == [first, second]
        assert stopped == [first]
        assert feeds.runner("Same") is not None
        feeds.stop("Same")
        assert stopped == [first, second]
        assert feeds.runner("Same") is None
        assert feeds.status("Same") is None

    def test_names_and_case_insensitive_find(self):
        feeds = Feeds.get_instance()
        runner = Mock(name="iMessage")
        feeds._runners["iMessage"] = runner
        assert feeds.names() == ["iMessage"]
        assert feeds.find("imessage") is runner
        assert feeds.find(" IMESSAGE ") is runner
        assert feeds.find("Telegram") is None

    def test_find_also_matches_the_config_slug(self):
        """`//feed pause photos` reaches the feed whose history source is "iCloud Photos"."""
        feeds = Feeds.get_instance()
        runner = Mock(name="photos")
        runner._feed = Mock(feed_name="iCloud Photos", config_slug="photos")
        feeds._runners["iCloud Photos"] = runner
        assert feeds.find("photos") is runner
        assert feeds.find("icloud photos") is runner
        assert feeds.find("Photos ") is runner
        assert feeds.find("imessage") is None

@pytest.mark.unit
class TestDatabase:
    def test_feed_row_round_trip(self, floxy_db):
        assert floxy_db.get_feed("X") is None
        floxy_db.save_feed("X", cursor="5", horizon=1.0)
        floxy_db.save_feed("X", last_entry_ts=3.0)
        row = floxy_db.get_feed("X")
        assert (row["cursor"], row["horizon"], row["last_entry_ts"], row["live_since"]) == ("5", 1.0, 3.0, None)
        assert floxy_db.delete_feed("X") is True
        assert floxy_db.delete_feed("X") is False
        assert floxy_db.get_feed("X") is None

    def test_gap_columns_are_added_to_an_older_feeds_table(self, floxy_db):
        import sqlite3
        conn = sqlite3.connect(floxy_db.db_path)
        conn.execute("ALTER TABLE feeds DROP COLUMN gap_until")
        conn.execute("ALTER TABLE feeds DROP COLUMN resume_cursor")
        conn.execute("ALTER TABLE feeds DROP COLUMN gap_covered_until")
        conn.commit()
        conn.close()
        floxy_db._init_database()
        floxy_db.save_feed("X", cursor="1", gap_until=2.0, resume_cursor="9", gap_covered_until=3.0)
        row = floxy_db.get_feed("X")
        assert (row["gap_until"], row["resume_cursor"], row["gap_covered_until"]) == (2.0, "9", 3.0)

    def test_the_old_counter_columns_are_dropped_from_an_older_feeds_table(self, floxy_db):
        import sqlite3
        conn = sqlite3.connect(floxy_db.db_path)
        conn.execute("ALTER TABLE feeds ADD COLUMN entries INTEGER NOT NULL DEFAULT 0")
        conn.execute("ALTER TABLE feeds ADD COLUMN total INTEGER")
        conn.commit()
        conn.close()
        floxy_db._init_database()
        floxy_db.save_feed("X", cursor="1")
        assert not {"entries", "total"} & set(floxy_db.get_feed("X"))

    def test_save_feed_rejects_unknown_columns(self, floxy_db):
        with pytest.raises(ValueError):
            floxy_db.save_feed("X", colour="blue")
        with pytest.raises(ValueError):
            floxy_db.save_feed("X", entries=3)

    def test_count_history_entries_keeps_a_feeds_rows_apart_from_chat_turns_under_the_same_source(self, floxy_db):
        """The Telegram feed stores under "Telegram", and so do the Telegram chat's own turns."""
        seed_rows(floxy_db, 4, source="Telegram")
        seed_rows(floxy_db, 3, source="Telegram", role="Marcus", start=100)
        floxy_db.add_history_entry("assistant", "hi", 200.0, "Telegram")
        seed_rows(floxy_db, 2, source="iMessage", start=300)
        assert floxy_db.count_history_entries("Telegram", FEED_ROLE) == 4
        assert floxy_db.count_history_entries("iMessage", FEED_ROLE) == 2
        assert floxy_db.count_history_entries("Indigo", FEED_ROLE) == 0

    def test_history_timestamps_names_the_moments_the_source_has_a_row_at(self, floxy_db):
        seed_at(floxy_db, [10.0, 20.0, 30.0])
        seed_at(floxy_db, [40.0], source="Other")
        floxy_db.add_history_entry("Marcus", "a chat turn", 50.0, "Scripted")
        found = floxy_db.history_timestamps("Scripted", [10.0, 20.0, 40.0, 50.0, 60.0])
        assert found == {10.0, 20.0, 50.0}, "any role of the source counts, another source does not"
        assert floxy_db.history_timestamps("Scripted", []) == set()

    def test_history_timestamps_spans_the_sqlite_parameter_chunks(self, floxy_db):
        timestamps = [1000.0 + i for i in range(1200)]
        seed_at(floxy_db, timestamps[::2])
        found = floxy_db.history_timestamps("Scripted", timestamps)
        assert found == set(timestamps[::2])

    def test_batch_insert_reports_only_new_rows(self, floxy_db):
        rows = [("system", "a", 1.0, None), ("system", "b", 2.0, "home")]
        assert floxy_db.add_history_entries(rows, "iMessage") == 2
        assert floxy_db.add_history_entries(rows + [("system", "c", 3.0, None)], "iMessage") == 1
        assert floxy_db.add_history_entries([], "iMessage") == 0
        stored = floxy_db.get_history_range(0, 10)
        assert [(r["content"], r["source"], r["location"]) for r in stored] == \
               [("a", "iMessage", None), ("b", "iMessage", "home"), ("c", "iMessage", None)]
