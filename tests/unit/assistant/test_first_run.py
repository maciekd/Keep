"""FirstRun's pipeline state and the thread that moves it: SETUP → INDEXING → CURATING →
EXPANDING → DONE, armed only on a fresh install."""
import json
import time
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from assistant import FirstRun
from assistant.FirstRun import (CURATION_STEP_SECONDS, FULL_PHOTOS_BACKFILL_MONTHS, HISTORY_SOURCE, IMESSAGE_FEED,
                                PHOTOS_BACKFILL_KEY, PHOTOS_FEED, PLACEHOLDER_HEADING, CurationProgress, Pipeline,
                                Stage, State)
from assistant.integration.FeedRunner import FeedState


# --- State ---------------------------------------------------------------------------------

class TestState:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "first_run.json"
        state = State(stage=Stage.CURATING, started_at=1.0, entered_at=2.0,
                      curation_start="2026-08-13", curation_end="2026-09-14", chunks_done=3, chunks_total=9)
        state.save(path)
        assert State.load(path) == state
        assert json.loads(path.read_text())["stage"] == "curating"

    def test_missing_file_is_none(self, tmp_path):
        assert State.load(tmp_path / "nope.json") is None

    @pytest.mark.parametrize("content", ["not json", '{"stage": "flying", "started_at": 1, "entered_at": 1}',
                                         '{"stage": "setup"}', '{"stage": "setup", "started_at": 1, "entered_at": 1, "extra": 1}'])
    def test_unreadable_file_is_none(self, tmp_path, content):
        path = tmp_path / "first_run.json"
        path.write_text(content)
        assert State.load(path) is None

    def test_install_age_is_days_since_the_first_start_and_none_without_a_file(self, tmp_path):
        path = tmp_path / "first_run.json"
        State(stage=Stage.SETUP, started_at=1_000_000.0, entered_at=1_000_000.0).save(path)
        assert FirstRun.install_age_days(now=1_000_000.0 + 2.5 * 86400, state_path=path) == pytest.approx(2.5)
        assert FirstRun.install_age_days(now=1_000_000.0, state_path=tmp_path / "nope.json") is None

    def test_span(self):
        state = State.fresh()
        assert state.span is None
        state.curation_start, state.curation_end = "2026-08-13", "2026-09-14"
        assert state.span == (date(2026, 8, 13), date(2026, 9, 14))

# --- arming ---------------------------------------------------------------------------------

class TestBeginIfFresh:
    @pytest.fixture(autouse=True)
    def a_users_install(self, monkeypatch):
        """The install's own `is_developer` decides nothing here: on the maintainer's machine it is
        on, and arming then set FullDiskAccess's process-wide pretence for real — every later
        probe in the same worker answered "not granted"."""
        monkeypatch.setattr(FirstRun, "config", SimpleNamespace(get=lambda key, default=None: default))
        monkeypatch.setattr(FirstRun.FullDiskAccess, "pretend_missing",
                            lambda on=True: pytest.fail("a user's install pretended the grant missing"))

    def test_seeded_stub_arms_the_pipeline(self, tmp_path, monkeypatch):
        monkeypatch.setattr(FirstRun.MemoryFile, "read", lambda *a, **k: "# Real bank\nfacts")
        path = tmp_path / "first_run.json"
        assert FirstRun.begin_if_fresh(True, path) is True
        assert State.load(path).stage == Stage.SETUP

    def test_placeholder_bank_arms_the_pipeline(self, tmp_path, monkeypatch):
        monkeypatch.setattr(FirstRun.MemoryFile, "read", lambda *a, **k: PLACEHOLDER_HEADING + "\n\nnothing yet")
        path = tmp_path / "first_run.json"
        assert FirstRun.begin_if_fresh(False, path) is True
        assert State.load(path).stage == Stage.SETUP

    def test_established_install_is_left_alone(self, tmp_path, monkeypatch):
        monkeypatch.setattr(FirstRun.MemoryFile, "read", lambda *a, **k: "# Long-term memory bank\n\n## Family")
        path = tmp_path / "first_run.json"
        assert FirstRun.begin_if_fresh(False, path) is False
        assert not path.exists()

    def test_existing_state_is_kept(self, tmp_path, monkeypatch):
        monkeypatch.setattr(FirstRun.MemoryFile, "read", lambda *a, **k: PLACEHOLDER_HEADING)
        path = tmp_path / "first_run.json"
        state = State(stage=Stage.CURATING, started_at=1.0, entered_at=2.0, chunks_done=2, chunks_total=8)
        state.save(path)
        assert FirstRun.begin_if_fresh(True, path) is True
        assert State.load(path) == state

    def test_unreadable_state_counts_as_not_armed(self, tmp_path):
        path = tmp_path / "first_run.json"
        path.write_text("garbage")
        assert FirstRun.begin_if_fresh(True, path) is False

    @pytest.fixture
    def developer(self, monkeypatch):
        knobs = SimpleNamespace(is_developer=True, pretended=[])
        monkeypatch.setattr(FirstRun, "config", SimpleNamespace(
            get=lambda key, default=None: knobs.is_developer if key == "is_developer" else default))
        monkeypatch.setattr(FirstRun.FullDiskAccess, "pretend_missing", lambda on=True: knobs.pretended.append(on))
        return knobs

    def test_a_developers_fresh_install_pretends_the_grant_is_missing(self, tmp_path, developer):
        # The developer's Mac has the grant, so the step would otherwise never be exercised.
        assert FirstRun.begin_if_fresh(True, tmp_path / "first_run.json") is True
        assert developer.pretended == [True]

    def test_a_users_fresh_install_probes_for_real(self, tmp_path, developer):
        developer.is_developer = False
        assert FirstRun.begin_if_fresh(True, tmp_path / "first_run.json") is True
        assert developer.pretended == []

    def test_the_restarted_developer_process_probes_for_real(self, tmp_path, developer):
        # The state file is there from the first process: this is the restart the step ends with.
        path = tmp_path / "first_run.json"
        State.fresh().save(path)
        assert FirstRun.begin_if_fresh(False, path) is True
        assert developer.pretended == []


class TestStageQueries:
    @pytest.mark.parametrize("stage, pending", [(Stage.SETUP, False), (Stage.INDEXING, True), (Stage.CURATING, False),
                                                (Stage.EXPANDING, False), (Stage.DONE, False)])
    def test_curation_pending_only_while_indexing_for_the_first_bank(self, tmp_path, stage, pending):
        """SETUP does not hold the nightly run back: an install that never enables Photos stays
        there for good and would otherwise never get a memory."""
        path = tmp_path / "first_run.json"
        state = State.fresh()
        state.enter(stage)
        state.save(path)
        assert FirstRun.stage(path) == stage
        assert FirstRun.curation_pending(path) is pending

    def test_a_stalled_catch_up_stops_holding_the_nightly_curation_back(self, tmp_path):
        path = tmp_path / "first_run.json"
        state = State.fresh(now=0)
        state.enter(Stage.INDEXING, now=0)
        state.save(path)
        wait = FirstRun.INDEXING_CURATION_WAIT_DAYS * 86400
        assert FirstRun.curation_pending(path, now=wait - 60) is True
        assert FirstRun.curation_pending(path, now=wait + 60) is False

    def test_no_file_means_no_stage_and_nothing_pending(self, tmp_path):
        path = tmp_path / "nope.json"
        assert FirstRun.stage(path) is None
        assert FirstRun.curation_pending(path) is False


# --- the pipeline -----------------------------------------------------------------------------

class _Runner:
    def __init__(self, state=FeedState.CATCHING_UP, live_since=None, stored=0, horizon_ts=0.0):
        self.state = state
        self._live_since = live_since
        self._stored = stored
        self._horizon_ts = horizon_ts

    def status(self):
        catching = self.state == FeedState.CATCHING_UP
        return {"state": self.state.value, "paused": False, "live_since": self._live_since, "stored": self._stored,
                "horizon_ts": self._horizon_ts, "progress": 0.5 if catching else None,
                "remaining": 10 if catching else 0, "last_error": None}


class _Feeds:
    def __init__(self, **runners):
        self._runners = runners

    def runner(self, name):
        return self._runners.get(name)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A pipeline with a tmp state file, fakes for everything it reaches out to, and knobs to
    turn: `enabled`, `feeds`, `feed_rows` (the feeds table), `backfill_months`."""
    knobs = SimpleNamespace(enabled=["Photos", "iMessage Feed"], feeds=_Feeds(), feed_rows={}, backfill_months=1,
                            history=Mock(), path=tmp_path / "first_run.json")
    monkeypatch.setattr(FirstRun, "config", SimpleNamespace(
        enabled_integrations=lambda: list(knobs.enabled),
        get=lambda key, default=None: knobs.backfill_months if key == PHOTOS_BACKFILL_KEY else default))
    monkeypatch.setattr(FirstRun.FloxyDatabase, "get_instance",
                        classmethod(lambda cls: SimpleNamespace(get_feed=lambda name: knobs.feed_rows.get(name))))
    knobs.pipeline = Pipeline(knobs.history, poll_seconds=0.01, feeds=knobs.feeds, path=knobs.path)

    def state_at(stage, **fields):
        state = State.fresh(now=time.time())
        state.enter(stage)
        for k, v in fields.items():
            setattr(state, k, v)
        state.save(knobs.path)
        return state
    knobs.state_at = state_at
    return knobs


def _rows(history):
    return [c.args[1] for c in history.add_context_message.call_args_list]


class TestCurationProgress:
    def test_moves_through_the_first_chunk_at_the_default_pace(self):
        p = CurationProgress.begin(chunks_total=4, chunks_done=0, now=1000.0)
        assert p.step_seconds == CURATION_STEP_SECONDS
        assert p.fraction(1000.0) == 0.0
        assert p.fraction(1000.0 + CURATION_STEP_SECONDS / 2) == pytest.approx(0.125)

    def test_a_slow_chunk_waits_at_its_boundary(self):
        p = CurationProgress.begin(chunks_total=4, chunks_done=0, now=1000.0)
        assert p.fraction(1000.0 + 10 * CURATION_STEP_SECONDS) == pytest.approx(0.25)

    def test_the_pace_is_the_average_of_the_chunks_this_process_finished(self):
        p = CurationProgress.begin(chunks_total=4, chunks_done=0, now=1000.0)
        p = p.chunk_done(1, now=1100.0).chunk_done(2, now=1300.0)
        assert p.step_seconds == 150.0
        assert p.fraction(1300.0) == pytest.approx(0.5)
        assert p.fraction(1375.0) == pytest.approx(0.625)

    def test_a_resumed_span_counts_only_its_own_chunks_for_the_pace(self):
        p = CurationProgress.begin(chunks_total=9, chunks_done=6, now=0.0)
        assert p.fraction(0.0) == pytest.approx(6 / 9)
        p = p.chunk_done(7, now=90.0)
        assert p.step_seconds == 90.0
        assert p.fraction(135.0) == pytest.approx(7.5 / 9)

    def test_eta_is_the_chunks_left_at_the_pace_less_the_current_elapsed(self):
        p = CurationProgress.begin(chunks_total=4, chunks_done=0, now=1000.0)
        assert p.eta_seconds(1000.0) == 4 * CURATION_STEP_SECONDS
        assert p.eta_seconds(1040.0) == 4 * CURATION_STEP_SECONDS - 40
        p = p.chunk_done(1, now=1100.0).chunk_done(2, now=1300.0)     # 150 s per chunk
        assert p.eta_seconds(1300.0) == 300.0
        assert p.eta_seconds(1375.0) == 225.0
        assert p.eta_seconds(1e6) == 150.0, "a slow chunk keeps the ETA at the chunks after it"
        assert CurationProgress.begin(chunks_total=2, chunks_done=2, now=0.0).eta_seconds(0.0) == 0.0

    def test_never_past_the_end(self):
        p = CurationProgress.begin(chunks_total=2, chunks_done=2, now=0.0)
        assert p.fraction(1e6) == 1.0
        assert CurationProgress.begin(chunks_total=0, chunks_done=0, now=0.0).fraction(5.0) == 0.0


@pytest.mark.usefixtures("on_a_mac")
class TestSetup:
    def test_waits_for_photos(self, env):
        env.enabled = ["iMessage Feed", "iCloud"]
        state = env.state_at(Stage.SETUP)
        assert env.pipeline.tick(state) is False
        assert State.load(env.path).stage == Stage.SETUP
        env.history.add_context_message.assert_not_called()

    def test_advances_when_photos_is_enabled(self, env):
        env.enabled = ["imessage", "photos"]   # spelling as the user wrote it
        state = env.state_at(Stage.SETUP)
        assert env.pipeline.tick(state) is True
        assert State.load(env.path).stage == Stage.INDEXING
        assert _rows(env.history) == ["Photos is set up. Reading the last month of photos and the message history, "
                                      "the groundwork for a long-term memory of the household."]
        assert env.history.add_context_message.call_args.kwargs["source"] == HISTORY_SOURCE


class TestIndexing:
    HORIZON = datetime(2026, 8, 13, 9, 30).timestamp()

    def test_waits_while_photos_has_no_runner(self, env):
        env.feeds = _Feeds(**{IMESSAGE_FEED: _Runner(FeedState.LIVE)})
        env.pipeline._feeds = env.feeds
        assert env.pipeline.tick(env.state_at(Stage.INDEXING)) is False
        assert State.load(env.path).stage == Stage.INDEXING

    def test_waits_while_photos_catches_up(self, env):
        env.pipeline._feeds = _Feeds(**{PHOTOS_FEED: _Runner(FeedState.CATCHING_UP), IMESSAGE_FEED: _Runner(FeedState.LIVE)})
        assert env.pipeline.tick(env.state_at(Stage.INDEXING)) is False

    def test_waits_for_imessage_when_it_is_enabled(self, env):
        env.pipeline._feeds = _Feeds(**{PHOTOS_FEED: _Runner(FeedState.LIVE), IMESSAGE_FEED: _Runner(FeedState.CATCHING_UP)})
        assert env.pipeline.tick(env.state_at(Stage.INDEXING)) is False
        env.pipeline._feeds = _Feeds(**{PHOTOS_FEED: _Runner(FeedState.LIVE)})   # not even started
        assert env.pipeline.tick(env.state_at(Stage.INDEXING)) is False

    def test_photos_alone_suffices_without_imessage(self, env):
        env.enabled = ["Photos", "iCloud"]
        env.feed_rows[PHOTOS_FEED] = {"horizon": self.HORIZON}
        env.pipeline._feeds = _Feeds(**{PHOTOS_FEED: _Runner(FeedState.LIVE)})
        assert env.pipeline.tick(env.state_at(Stage.INDEXING)) is True
        assert State.load(env.path).stage == Stage.CURATING

    def test_fixes_the_span_from_the_photos_horizon_and_notifies(self, env):
        from assistant.agents.LongTermMemoryCurator import LongTermMemoryCurator
        env.feed_rows[PHOTOS_FEED] = {"horizon": self.HORIZON}
        env.pipeline._feeds = _Feeds(**{PHOTOS_FEED: _Runner(FeedState.LIVE), IMESSAGE_FEED: _Runner(FeedState.LIVE)})
        state = env.state_at(Stage.INDEXING)
        assert env.pipeline.tick(state) is True
        saved = State.load(env.path)
        assert saved.stage == Stage.CURATING
        assert saved.span == (date(2026, 8, 13), date.today())
        assert saved.chunks_done == 0
        assert saved.chunks_total == len(LongTermMemoryCurator.chunks(date(2026, 8, 13), date.today()))
        assert _rows(env.history)[0].startswith("Photos and messages are in. Building the first long-term memory "
                                                "from what happened between 13 August and ")

    def test_horizon_falls_back_to_the_runner(self, env):
        env.pipeline._feeds = _Feeds(**{PHOTOS_FEED: _Runner(FeedState.LIVE, horizon_ts=self.HORIZON),
                                        IMESSAGE_FEED: _Runner(FeedState.LIVE)})
        assert env.pipeline.tick(env.state_at(Stage.INDEXING)) is True
        assert State.load(env.path).span == (date(2026, 8, 13), date.today())


class TestCurating:
    SPAN = {"curation_start": "2026-08-13", "curation_end": "2026-09-14", "chunks_total": 9}

    def test_publishes_the_progress_while_the_span_runs(self, env, monkeypatch):
        seen = []

        def run_span(start, end, history=None, on_chunk=None, first_chunk=0, **_):
            seen.append(FirstRun.curation_progress())
            on_chunk(first_chunk + 1, 9, {})
            seen.append(FirstRun.curation_progress())
            return {'chunks': 9, 'curated': 9, 'empty': 0, 'failed': 0, 'interaction_count': 1}

        monkeypatch.setattr("assistant.agents.LongTermMemoryCurator.LongTermMemoryCurator.run_span", run_span)
        monkeypatch.setattr("assistant.agents.LongTermMemoryCurator.LongTermMemoryCurator.chunks",
                            lambda start, end: [None] * 9)
        env.pipeline._expand = lambda state, summary=None: True
        assert FirstRun.curation_progress() is None
        env.pipeline.tick(env.state_at(Stage.CURATING, chunks_done=2, **self.SPAN))
        assert [(p.chunks_done, p.chunks_total, p.resumed_at) for p in seen] == [(2, 9, 2), (3, 9, 2)]
        assert FirstRun.curation_progress() is None, "cleared once the span is over"

    def test_progress_is_cleared_when_the_span_fails(self, env, monkeypatch):
        def run_span(*_, **__):
            raise RuntimeError("boom")

        monkeypatch.setattr("assistant.agents.LongTermMemoryCurator.LongTermMemoryCurator.run_span", run_span)
        monkeypatch.setattr("assistant.agents.LongTermMemoryCurator.LongTermMemoryCurator.chunks",
                            lambda start, end: [None] * 9)
        with pytest.raises(RuntimeError):
            env.pipeline.tick(env.state_at(Stage.CURATING, chunks_done=2, **self.SPAN))
        assert FirstRun.curation_progress() is None

    def test_runs_the_span_and_persists_the_chunk_progress(self, env, monkeypatch):
        from assistant.agents.LongTermMemoryCurator import LongTermMemoryCurator
        calls = []

        def run_span(start, end, history=None, on_chunk=None, first_chunk=0, **_):
            calls.append((start, end, history, first_chunk))
            on_chunk(first_chunk + 1, 9, {})
            assert State.load(env.path).chunks_done == first_chunk + 1
            on_chunk(9, 9, {})
            return {'chunks': 9, 'curated': 7, 'empty': 1, 'failed': 1, 'interaction_count': 100}
        monkeypatch.setattr(LongTermMemoryCurator, "run_span", staticmethod(run_span))
        monkeypatch.setattr(env.pipeline, "_expand", lambda state, summary=None: True)
        state = env.state_at(Stage.CURATING, chunks_done=2, **self.SPAN)
        assert env.pipeline.tick(state) is True
        assert calls == [(date(2026, 8, 13), date(2026, 9, 14), env.history, 2)]
        assert State.load(env.path).chunks_done == 9

    def test_a_busy_curator_leaves_the_stage_for_the_next_poll(self, env, monkeypatch):
        from assistant.agents.LongTermMemoryCurator import LongTermMemoryCurator
        monkeypatch.setattr(LongTermMemoryCurator, "run_span", staticmethod(lambda *a, **k: None))
        state = env.state_at(Stage.CURATING, **self.SPAN)
        assert env.pipeline.tick(state) is False
        assert State.load(env.path).stage == Stage.CURATING

    def test_a_summary_leads_to_the_expansion(self, env, monkeypatch):
        from assistant.agents.LongTermMemoryCurator import LongTermMemoryCurator
        summary = {'chunks': 9, 'curated': 9, 'empty': 0, 'failed': 0, 'interaction_count': 1}
        monkeypatch.setattr(LongTermMemoryCurator, "run_span", staticmethod(lambda *a, **k: summary))
        expanded = []
        monkeypatch.setattr(env.pipeline, "_expand", lambda state, s=None: expanded.append(s) or True)
        assert env.pipeline.tick(env.state_at(Stage.CURATING, **self.SPAN)) is True
        assert expanded == [summary]

    def test_missing_span_skips_straight_to_the_expansion(self, env, monkeypatch):
        expanded = []
        monkeypatch.setattr(env.pipeline, "_expand", lambda state, s=None: expanded.append(s) or True)
        assert env.pipeline.tick(env.state_at(Stage.CURATING)) is True
        assert expanded == [None]

    def _span_of(self, monkeypatch, env, on_run):
        """Patch run_span with a fake that drives `on_chunk` through `on_run(on_chunk, first_chunk)` and
        returns its summary; `_expand` records the summaries it gets."""
        from assistant.agents.LongTermMemoryCurator import LongTermMemoryCurator
        expanded = []

        def run_span(start, end, history=None, on_chunk=None, first_chunk=0, **_):
            return on_run(on_chunk, first_chunk)
        monkeypatch.setattr(LongTermMemoryCurator, "run_span", staticmethod(run_span))
        monkeypatch.setattr(env.pipeline, "_expand", lambda state, s=None: expanded.append(s) or True)
        return expanded

    def test_a_reset_during_the_span_stops_it_and_saves_nothing(self, env, monkeypatch):
        verdicts = []

        def run(on_chunk, first_chunk):
            verdicts.append(on_chunk(3, 9, {}))
            env.path.unlink()                       # //first_run reset
            verdicts.append(on_chunk(4, 9, {}))
            return {'chunks': 9, 'curated': 4, 'empty': 0, 'failed': 0, 'aborted': True, 'interaction_count': 1}
        expanded = self._span_of(monkeypatch, env, run)
        assert env.pipeline.tick(env.state_at(Stage.CURATING, chunks_done=2, **self.SPAN)) is False
        assert verdicts == [True, False]
        assert not env.path.exists(), "the chunk save must not recreate the deleted state"
        assert expanded == []

    def test_a_stage_forced_during_the_span_stops_it_and_keeps_the_forced_stage(self, env, monkeypatch):
        def run(on_chunk, first_chunk):
            forced = State.load(env.path)
            forced.enter(Stage.INDEXING)
            forced.save(env.path)                   # //first_run stage indexing
            assert on_chunk(3, 9, {}) is False
            return {'chunks': 9, 'curated': 3, 'empty': 0, 'failed': 0, 'aborted': True, 'interaction_count': 1}
        expanded = self._span_of(monkeypatch, env, run)
        assert env.pipeline.tick(env.state_at(Stage.CURATING, chunks_done=2, **self.SPAN)) is False
        saved = State.load(env.path)
        assert saved.stage == Stage.INDEXING and saved.chunks_done == 2
        assert expanded == []

    def test_an_all_failed_span_is_not_built_and_is_retried_later(self, env, monkeypatch):
        def run(on_chunk, first_chunk):
            for i in range(first_chunk + 1, 10):
                on_chunk(i, 9, None)
            return {'chunks': 9, 'curated': 0, 'empty': 0, 'failed': 9 - first_chunk, 'aborted': False,
                    'interaction_count': 0}
        expanded = self._span_of(monkeypatch, env, run)
        clock = [1000.0]
        monkeypatch.setattr(FirstRun.time, "monotonic", lambda: clock[0])
        state = env.state_at(Stage.CURATING, chunks_done=2, **self.SPAN)
        assert env.pipeline.tick(state) is False
        saved = State.load(env.path)
        assert saved.stage == Stage.CURATING and saved.chunks_done == 2, "rewound to where the span resumed"
        assert expanded == []
        # Not run again until the retry pause is over.
        runs = []
        self._span_of(monkeypatch, env, lambda on_chunk, first_chunk: runs.append(first_chunk) or
                      {'chunks': 9, 'curated': 7, 'empty': 0, 'failed': 0, 'aborted': False, 'interaction_count': 1})
        assert env.pipeline.tick(State.load(env.path)) is False and runs == []
        clock[0] += FirstRun.CURATION_RETRY_SECONDS
        assert env.pipeline.tick(State.load(env.path)) is True
        assert runs == [2]

    def test_a_partly_failed_span_still_counts_as_built(self, env, monkeypatch):
        summary = {'chunks': 9, 'curated': 1, 'empty': 0, 'failed': 6, 'aborted': False, 'interaction_count': 1}
        expanded = self._span_of(monkeypatch, env, lambda on_chunk, first_chunk: summary)
        assert env.pipeline.tick(env.state_at(Stage.CURATING, chunks_done=2, **self.SPAN)) is True
        assert expanded == [summary]


@pytest.fixture
def expansion(monkeypatch):
    """What `_expand` reaches for: the config store write and the integration restart."""
    import common.ConfigStore as store
    from assistant.integration.IntegrationMonitor import IntegrationMonitor
    calls = SimpleNamespace(set_value=[], restart=[])
    monkeypatch.setattr(store, "set_value", lambda key, raw: calls.set_value.append((key, raw)) or "ok")
    monkeypatch.setattr(IntegrationMonitor, "get_instance",
                        classmethod(lambda cls: SimpleNamespace(restart=lambda name: calls.restart.append(name) or {})))
    return calls


class TestExpand:
    SUMMARY = {'chunks': 9, 'curated': 8, 'empty': 1, 'failed': 0, 'interaction_count': 100}

    def test_widens_the_horizon_and_restarts_photos(self, env, expansion):
        state = env.state_at(Stage.CURATING)
        assert env.pipeline._expand(state, self.SUMMARY) is True
        assert expansion.set_value == [(PHOTOS_BACKFILL_KEY, str(FULL_PHOTOS_BACKFILL_MONTHS))]
        assert expansion.restart == ["Photos"]
        assert State.load(env.path).stage == Stage.EXPANDING
        assert _rows(env.history) == ["The first long-term memory is built. Now indexing "
                                      "the last 3 years of photos and videos in the background; with the memory in "
                                      "place their descriptions come out better."]

    def test_a_wider_horizon_set_by_hand_is_left_alone(self, env, expansion):
        env.backfill_months = 60
        state = env.state_at(Stage.CURATING)
        assert env.pipeline._expand(state, self.SUMMARY) is True
        assert expansion.set_value == [] and expansion.restart == []
        assert State.load(env.path).stage == Stage.DONE
        assert "already set to reach 60 months back" in _rows(env.history)[0]

    def test_an_unparsable_horizon_counts_as_the_default(self, env, expansion):
        env.backfill_months = "lots"
        env.pipeline._expand(env.state_at(Stage.CURATING), self.SUMMARY)
        assert expansion.set_value == [(PHOTOS_BACKFILL_KEY, "36")]

    def test_the_stage_is_stamped_before_the_restart(self, env, expansion, monkeypatch):
        """A small library goes live inside the synchronous restart: that `live_since` must count."""
        from assistant.integration.IntegrationMonitor import IntegrationMonitor
        went_live = []

        def restart(name):
            went_live.append(time.time())
            time.sleep(0.01)
            return {}
        monkeypatch.setattr(IntegrationMonitor, "get_instance", classmethod(lambda cls: SimpleNamespace(restart=restart)))
        env.pipeline._expand(env.state_at(Stage.CURATING), self.SUMMARY)
        state = State.load(env.path)
        assert state.stage == Stage.EXPANDING and state.entered_at <= went_live[0]
        env.pipeline._feeds = _Feeds(**{PHOTOS_FEED: _Runner(FeedState.LIVE, live_since=went_live[0], stored=5)})
        assert env.pipeline.tick(state) is True

    def test_a_missing_integration_still_advances(self, env, expansion, monkeypatch):
        from assistant.integration.IntegrationMonitor import IntegrationMonitor
        monkeypatch.setattr(IntegrationMonitor, "get_instance", classmethod(lambda cls: SimpleNamespace(restart=lambda n: None)))
        assert env.pipeline._expand(env.state_at(Stage.CURATING), self.SUMMARY) is True
        assert State.load(env.path).stage == Stage.EXPANDING


class TestExpanding:
    def test_waits_while_the_wider_index_catches_up(self, env):
        state = env.state_at(Stage.EXPANDING)
        env.pipeline._feeds = _Feeds(**{PHOTOS_FEED: _Runner(FeedState.CATCHING_UP)})
        assert env.pipeline.tick(state) is False
        env.pipeline._feeds = _Feeds()
        assert env.pipeline.tick(state) is False
        assert State.load(env.path).stage == Stage.EXPANDING

    def test_a_live_since_before_the_restart_does_not_count(self, env):
        state = env.state_at(Stage.EXPANDING)
        env.pipeline._feeds = _Feeds(**{PHOTOS_FEED: _Runner(FeedState.LIVE, live_since=state.entered_at - 60)})
        assert env.pipeline.tick(state) is False
        env.pipeline._feeds = _Feeds(**{PHOTOS_FEED: _Runner(FeedState.LIVE, live_since=None)})
        assert env.pipeline.tick(state) is False

    def test_done_once_the_feed_is_live_again(self, env):
        state = env.state_at(Stage.EXPANDING)
        env.pipeline._feeds = _Feeds(**{PHOTOS_FEED: _Runner(FeedState.LIVE, live_since=state.entered_at + 5, stored=4321)})
        assert env.pipeline.tick(state) is True
        assert State.load(env.path).stage == Stage.DONE
        assert _rows(env.history) == ["The photo library is indexed: 4321 photos and videos are searchable. "
                                      "The first-run setup is complete."]


@pytest.mark.usefixtures("on_a_mac")
class TestThread:
    def test_does_not_start_without_state(self, env):
        env.pipeline.start()
        assert not env.pipeline._thread.is_alive()

    def test_run_keeps_polling_when_done_so_a_forced_stage_revives_it(self, env):
        # //first_run stage <name> after DONE (or after a reset) must still be picked up
        env.state_at(Stage.DONE)
        env.pipeline.start()
        time.sleep(0.05)
        assert env.pipeline._thread.is_alive()
        env.state_at(Stage.SETUP)
        deadline = time.time() + 5
        while State.load(env.path).stage != Stage.INDEXING and time.time() < deadline:
            time.sleep(0.01)
        assert State.load(env.path).stage == Stage.INDEXING
        env.pipeline.stop()

    def test_run_moves_through_a_stage_and_stops_on_request(self, env):
        env.state_at(Stage.SETUP)
        env.pipeline.start()
        deadline = time.time() + 5
        while State.load(env.path).stage != Stage.INDEXING and time.time() < deadline:
            time.sleep(0.01)
        assert State.load(env.path).stage == Stage.INDEXING
        env.pipeline.stop()
        assert not env.pipeline._thread.is_alive()

    def test_run_idles_when_the_state_file_disappears(self, env):
        env.enabled = []
        env.state_at(Stage.SETUP)
        env.pipeline.start()
        env.path.unlink()
        time.sleep(0.05)
        assert env.pipeline._thread.is_alive()
        env.state_at(Stage.SETUP)
        env.enabled = ["Photos"]
        deadline = time.time() + 5
        while State.load(env.path).stage != Stage.INDEXING and time.time() < deadline:
            time.sleep(0.01)
        assert State.load(env.path).stage == Stage.INDEXING
        env.pipeline.stop()

    def test_a_failing_step_does_not_kill_the_thread(self, env, monkeypatch):
        env.state_at(Stage.SETUP)
        attempts = []

        def tick(state):
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("boom")
            state.enter(Stage.DONE)
            state.save(env.path)
            return True
        monkeypatch.setattr(env.pipeline, "tick", tick)
        env.pipeline.start()
        deadline = time.time() + 5
        while len(attempts) < 2 and time.time() < deadline:
            time.sleep(0.01)
        env.pipeline.stop()
        assert len(attempts) == 2 and State.load(env.path).stage == Stage.DONE


@pytest.mark.usefixtures("on_a_mac")
class TestSnappiness:
    def test_a_wait_reason_is_traced_once_until_it_changes(self, env, monkeypatch):
        from assistant import FirstRun
        traced = []
        monkeypatch.setattr(FirstRun, "trace", lambda msg: "saved" in msg or traced.append(msg))
        env.enabled = []
        env.state_at(Stage.SETUP)
        for _ in range(3):
            env.pipeline.tick(State.load(env.path))
        assert traced == ["First run: waiting for the Photos integration to be enabled"]
        env.enabled = ["Photos"]
        env.pipeline.tick(State.load(env.path))          # → INDEXING, and the Photos runner is missing
        env.pipeline.tick(State.load(env.path))
        assert len(traced) == 2 and traced[1].startswith("First run: still indexing")

    def test_a_transition_gets_its_next_tick_at_once(self, env, monkeypatch):
        waits = []
        monkeypatch.setattr(env.pipeline._stop_event, "wait", lambda seconds: waits.append(seconds) or env.pipeline.stop())
        env.enabled = ["Photos"]
        env.state_at(Stage.SETUP)
        env.pipeline._run()
        # SETUP → INDEXING ticked straight into INDEXING (no runner yet), only then waited
        assert State.load(env.path).stage == Stage.INDEXING and waits == [0.01]


@pytest.mark.usefixtures("on_a_mac")
class TestDesktopShortcut:
    """The shortcut is built the first time the Full Disk Access probe passes, once, and the
    state remembers the attempt."""

    @pytest.fixture
    def shortcut(self, env, monkeypatch):
        knobs = SimpleNamespace(granted=False, built=True, ensure_calls=0)

        def ensure():
            knobs.ensure_calls += 1
            return knobs.built
        monkeypatch.setattr(FirstRun.FullDiskAccess, "granted", lambda: knobs.granted)
        monkeypatch.setattr(FirstRun.DesktopShortcut, "ensure", ensure)
        knobs.autostart_failure, knobs.enable_calls = None, 0

        def enable():
            knobs.enable_calls += 1
            return knobs.autostart_failure
        monkeypatch.setattr(FirstRun.Autostart, "enable", enable)
        knobs.chat = Mock()
        knobs.chat.open_conversation.return_value = False     # no chat takes the turn unless a test says so
        monkeypatch.setattr(FirstRun, "notifier", lambda: knobs.chat)
        return knobs

    def test_waits_while_the_access_is_not_granted(self, env, shortcut):
        state = env.state_at(Stage.SETUP)
        assert env.pipeline.desktop_shortcut(state, now=0.0) is False
        assert shortcut.ensure_calls == 0
        assert State.load(env.path).desktop_shortcut_at is None

    def test_a_probe_that_finds_the_grant_missing_is_remembered_across_the_restart(self, env, shortcut):
        state = env.state_at(Stage.SETUP)
        assert env.pipeline.desktop_shortcut(state, now=0.0) is False
        assert State.load(env.path).full_disk_access_awaited is True

    def test_builds_once_the_awaited_access_is_granted_and_opens_the_conversation(self, env, shortcut):
        # The grant quit the terminal, so the user hears it from the assistant on Telegram: the
        # turn tells the assistant the facts, and the history gets that exchange, not a bare row.
        state = env.state_at(Stage.SETUP, full_disk_access_awaited=True)
        shortcut.granted = True
        shortcut.chat.open_conversation.return_value = True
        assert env.pipeline.desktop_shortcut(state, now=0.0) is True
        assert shortcut.ensure_calls == 1
        assert State.load(env.path).desktop_shortcut_at is not None
        turn = shortcut.chat.open_conversation.call_args.args[0]
        assert "Full Disk Access" in turn and "Desktop" in turn
        assert _rows(env.history) == []

    def test_without_a_chat_to_take_the_turn_a_history_row_tells_it(self, env, shortcut):
        state = env.state_at(Stage.SETUP, full_disk_access_awaited=True)
        shortcut.granted = True
        assert env.pipeline.desktop_shortcut(state, now=0.0) is True
        assert any("Desktop" in row for row in _rows(env.history))

    def test_a_grant_present_from_the_start_builds_quietly(self, env, shortcut):
        # A developer's wipe, a reinstall: the brief opens the conversation, and an "I'm back,
        # the access works" turn on top of it would be nonsense.
        state = env.state_at(Stage.SETUP)
        shortcut.granted = True
        shortcut.chat.open_conversation.return_value = True
        assert env.pipeline.desktop_shortcut(state, now=0.0) is True
        shortcut.chat.open_conversation.assert_not_called()
        assert any("Desktop" in row for row in _rows(env.history))

    def test_never_a_second_attempt_after_the_stamp(self, env, shortcut):
        state = env.state_at(Stage.SETUP, desktop_shortcut_at=1.0)
        shortcut.granted = True
        assert env.pipeline.desktop_shortcut(state, now=1000.0) is False
        assert shortcut.ensure_calls == 0

    def test_the_login_item_is_registered_with_the_shortcut_and_the_turn_says_so(self, env, shortcut):
        state = env.state_at(Stage.SETUP, full_disk_access_awaited=True)
        shortcut.granted = True
        shortcut.chat.open_conversation.return_value = True
        assert env.pipeline.desktop_shortcut(state, now=0.0) is True
        assert shortcut.enable_calls == 1
        turn = shortcut.chat.open_conversation.call_args.args[0]
        assert "login" in turn

    def test_a_login_item_that_could_not_be_made_leaves_the_shortcut_step_whole(self, env, shortcut):
        state = env.state_at(Stage.SETUP)
        shortcut.granted, shortcut.autostart_failure = True, "the user declined"
        assert env.pipeline.desktop_shortcut(state, now=0.0) is True
        assert State.load(env.path).desktop_shortcut_at is not None
        rows = _rows(env.history)
        assert any("Desktop" in row for row in rows) and not any("login" in row for row in rows)

    def test_a_failed_build_is_stamped_and_leaves_no_history_row(self, env, shortcut):
        state = env.state_at(Stage.SETUP)
        shortcut.granted, shortcut.built = True, False
        assert env.pipeline.desktop_shortcut(state, now=0.0) is False
        assert State.load(env.path).desktop_shortcut_at is not None
        assert _rows(env.history) == []
        shortcut.chat.open_conversation.assert_not_called()

    def test_the_thread_lets_the_interfaces_come_up_before_the_first_probe(self, env, shortcut):
        # Telegram is still connecting when the pipeline starts; the turn must follow the
        # startup note, not race it.
        env.state_at(Stage.DONE)
        shortcut.granted = True
        env.pipeline.start()
        time.sleep(0.05)
        env.pipeline.stop()
        assert shortcut.ensure_calls == 0
        assert env.pipeline._shortcut_probe_at > time.monotonic()

    def test_the_probe_is_rate_limited_between_polls(self, env, shortcut):
        state = env.state_at(Stage.SETUP)
        env.pipeline.desktop_shortcut(state, now=0.0)
        shortcut.granted = True
        assert env.pipeline.desktop_shortcut(state, now=1.0) is False
        assert env.pipeline.desktop_shortcut(state, now=FirstRun.DESKTOP_SHORTCUT_PROBE_SECONDS) is True
        shortcut.chat.open_conversation.assert_called_once()    # the first probe found the grant missing

    def test_a_done_install_still_gets_the_shortcut(self, env, shortcut):
        state = env.state_at(Stage.DONE)
        shortcut.granted = True
        assert env.pipeline.desktop_shortcut(state, now=0.0) is True

    def test_once_setup_is_over_the_shortcut_is_built_without_the_grant(self, env, shortcut):
        # #167: the grant serves iMessage alone; an install that finished its setup without it
        # must not probe chat.db every minute for the rest of its life, nor go without the shortcut.
        env.enabled = ["Telegram", "Photos", "Google"]     # the threshold reached, no setup command recorded
        state = env.state_at(Stage.SETUP)
        assert env.pipeline.desktop_shortcut(state, now=0.0) is True
        assert shortcut.ensure_calls == 1
        saved = State.load(env.path)
        assert saved.desktop_shortcut_at is not None
        assert env.pipeline.desktop_shortcut(saved, now=FirstRun.DESKTOP_SHORTCUT_PROBE_SECONDS) is False
        rows = _rows(env.history)
        assert any("Desktop" in row for row in rows) and not any("Full Disk Access" in row for row in rows)
        shortcut.chat.open_conversation.assert_not_called()

    def test_the_login_item_is_not_registered_without_the_grant(self, env, shortcut):
        env.enabled = ["Telegram", "Photos", "Google"]
        state = env.state_at(Stage.SETUP)
        assert env.pipeline.desktop_shortcut(state, now=0.0) is True
        assert shortcut.enable_calls == 0

    def test_a_setup_still_in_progress_keeps_waiting_for_the_grant(self, env, shortcut):
        env.enabled = ["Telegram", "Photos", "Google"]
        FirstRun.note_setup_activity()      # a setup command just ran: the user is mid-setup
        state = env.state_at(Stage.SETUP)
        assert env.pipeline.desktop_shortcut(state, now=0.0) is False
        assert shortcut.ensure_calls == 0 and State.load(env.path).desktop_shortcut_at is None

    def test_the_shortcut_wait_and_the_stage_wait_are_each_traced_once(self, env, shortcut, monkeypatch):
        # #167: the two waits alternate on every poll; with one memory between them both logged again each time.
        traced = []
        monkeypatch.setattr(FirstRun, "trace", lambda message: traced.append(message))
        env.enabled = []
        state = env.state_at(Stage.SETUP)
        for n in range(3):
            env.pipeline.desktop_shortcut(state, now=n * FirstRun.DESKTOP_SHORTCUT_PROBE_SECONDS)
            env.pipeline.tick(state)
        waits = [m for m in traced if "Desktop shortcut waits" in m or "Photos integration" in m]
        assert len(waits) == 2


class TestOnAServerThatIsNotAMac:
    """No Desktop shortcut, login item or Full Disk Access to wait for, and no Photos to index."""

    @pytest.fixture(autouse=True)
    def linux(self, monkeypatch):
        from common import Platform
        monkeypatch.setattr(Platform, "current", lambda: Platform.LINUX)

    @pytest.fixture
    def untouched(self, monkeypatch):
        def never(*args, **kwargs):
            raise AssertionError("a Mac-only step ran on Linux")
        for module, name in ((FirstRun.FullDiskAccess, "granted"), (FirstRun.DesktopShortcut, "ensure"),
                             (FirstRun.Autostart, "enable")):
            monkeypatch.setattr(module, name, never)
        chat = Mock()
        monkeypatch.setattr(FirstRun, "notifier", lambda: chat)
        return chat

    def test_the_shortcut_step_is_stamped_at_once_without_a_probe_a_turn_or_a_row(self, env, untouched):
        state = env.state_at(Stage.SETUP)
        assert env.pipeline.desktop_shortcut(state, now=0.0) is False
        saved = State.load(env.path)
        assert saved.desktop_shortcut_at is not None and not saved.full_disk_access_awaited
        untouched.open_conversation.assert_not_called()
        assert _rows(env.history) == []

    def test_a_config_copied_from_a_mac_does_not_send_the_pipeline_waiting_for_photos(self, env):
        assert "Photos" in env.enabled
        state = env.state_at(Stage.SETUP)
        assert env.pipeline.tick(state) is False
        assert State.load(env.path).stage == Stage.SETUP
