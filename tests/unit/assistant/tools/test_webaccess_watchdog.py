"""Browse watchdog hardening: the sync-side timeout cancels the coroutine (instead of
abandoning it to wedge the loop), the session lock serializes commands on the shared
tab, and single CDP evaluates are bounded. Motivated by the 2026-08-31 checkout hang:
a click into Shopify checkout waited 120s on an unanswered evaluate, the timeout left
the coroutine running, and the recovery browse_open interleaved with it and hung too."""

import asyncio
import threading

import pytest
from unittest.mock import Mock, patch, AsyncMock

from assistant.tools.WebAccess import WebAccess, BrowseWatchdogTimeout, _LoopThread
from tests.unit.waiting import FAILURE_BOUND


def make_wa(**overrides):
    wa = WebAccess.__new__(WebAccess)
    wa._browser = Mock()
    wa._tab = Mock()
    wa._chrome_pids = set()
    wa._driver_ready = threading.Event()
    wa._driver_ready.set()
    wa._last_init_failure_ts = 0.0
    wa._init_in_progress = False
    wa._browse_session_url = None
    wa._browse_snapshot_frame_ids = []
    wa._browse_snapshot_labels = []
    wa._browse_frame_sessions = {}
    wa._popup_parents = []
    wa._known_page_targets = set()
    wa._pending_target_notes = []
    wa._dialog_notes = []
    wa._pending_popup_targets = {}
    wa._browse_pdf_text = None
    wa._last_screenshot_bytes = None
    wa._last_screenshot_path = None
    wa._browse_fail_streak = 0
    wa._browser_hard_down = False
    wa._browser_closed_by_user = False
    wa._browser_generation = 1
    wa._browse_session_generation = 1
    wa._browse_lock = asyncio.Lock()
    for key, value in overrides.items():
        setattr(wa, key, value)
    return wa


@pytest.fixture
def loop_thread():
    lt = _LoopThread()
    lt.start()
    yield lt
    lt.stop()


@pytest.mark.unit
class TestSubmitWatchdog:

    def test_timeout_raises_watchdog_and_cancels_coroutine(self, loop_thread):
        cancelled = threading.Event()

        async def wedge():
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        with pytest.raises(BrowseWatchdogTimeout):
            loop_thread.submit(wedge(), timeout=0.05)
        assert cancelled.wait(timeout=FAILURE_BOUND), "watchdog did not cancel the abandoned coroutine"

    def test_coroutines_own_timeout_error_is_not_watchdog_expiry(self, loop_thread):
        # On 3.12 asyncio.TimeoutError IS the builtin TimeoutError that
        # fut.result(timeout) raises on expiry; an inner CDP timeout escaping the
        # coroutine must propagate as itself, not become a spurious watchdog
        # cancellation (session reset + "cancelled after Ns" fiction).
        async def inner_timeout():
            raise asyncio.TimeoutError("inner CDP wait expired")

        with pytest.raises(asyncio.TimeoutError):
            loop_thread.submit(inner_timeout(), timeout=FAILURE_BOUND)

    def test_watchdog_exceeds_the_sum_of_stage_budgets(self):
        # Every stage inside a browse command is sanctioned to run this long; a
        # slow-but-healthy page working through its soft caps must never trip
        # the whole-command watchdog. Raise _BROWSE_OP_TIMEOUT if a stage cap grows.
        worst_case = (WebAccess._NAV_COMMIT_TIMEOUT + WebAccess._PAGE_READY_TIMEOUT
                      + WebAccess._PAGE_QUIET_TIMEOUT
                      + WebAccess._COOKIE_RETRY_DELAY + 2 * WebAccess._EVAL_TIMEOUT
                      + WebAccess._SNAPSHOT_FRAME_BUDGET
                      + WebAccess._INIT_INTER_RETRY_DELAY)
        assert WebAccess._BROWSE_OP_TIMEOUT > worst_case


@pytest.mark.unit
class TestBrowseLock:

    def test_commands_serialize_on_the_lock(self, loop_thread):
        wa = make_wa(_loop_thread=loop_thread)
        order = []
        first_started = threading.Event()

        async def first():
            order.append("first-start")
            first_started.set()
            await asyncio.sleep(0.05)
            order.append("first-end")
            return "a"

        async def second():
            order.append("second-start")
            return "b"

        t = threading.Thread(
            target=lambda: loop_thread.submit(wa._with_browse_lock(first()), timeout=FAILURE_BOUND))
        t.start()
        assert first_started.wait(timeout=FAILURE_BOUND)  # first() holds the lock before second() is submitted
        result = loop_thread.submit(wa._with_browse_lock(second()), timeout=FAILURE_BOUND)
        t.join()

        assert result == "b"
        assert order == ["first-start", "first-end", "second-start"]

@pytest.mark.unit
class TestRunBrowseOp:

    def test_watchdog_expiry_resets_session_and_reports(self, loop_thread):
        wa = make_wa(_loop_thread=loop_thread,
                     _browse_snapshot_frame_ids=["stale"],
                     _browse_snapshot_labels=["stale"])

        async def wedge():
            await asyncio.sleep(30)

        with patch.object(WebAccess, '_BROWSE_OP_TIMEOUT', 0.05), \
                patch.object(WebAccess, '_async_browser_alive', new=AsyncMock(return_value=True)):
            result = wa._run_browse_op("browse_act(55)", wedge(), "ERROR: action failed.")

        assert "browse_act(55)" in result
        assert "cancelled" in result
        assert "browse_open" in result  # tells the LLM how to resume
        assert wa._browse_snapshot_frame_ids == []  # session state was reset

    def test_other_exceptions_return_the_generic_error(self, loop_thread):
        wa = make_wa(_loop_thread=loop_thread)

        async def boom():
            raise RuntimeError("nope")

        result = wa._run_browse_op("browse_act(1)", boom(), "ERROR: action failed.")
        assert result == "ERROR: action failed."


@pytest.mark.unit
class TestBoundedEvaluate:

    def test_evaluate_times_out_instead_of_hanging(self):
        tab = Mock()
        tab.evaluate = Mock(side_effect=lambda *a, **k: asyncio.sleep(30))

        async def go():
            await WebAccess._evaluate(tab, "1+1", timeout=0.05)

        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(go())

@pytest.mark.unit
class TestSnapshotFrameBudget:

    def test_slow_frames_are_skipped_once_budget_is_spent(self):
        wa = make_wa()
        wa._tab.target.url = "https://example.com"

        t1, t2 = Mock(), Mock()
        t1.url = "https://frames.example/one"
        t1.target_id = "t1"
        t2.url = "https://frames.example/two"
        t2.target_id = "t2"
        visited = []

        async def slow_frame_eval(_self, target_id, _expr):
            visited.append(str(target_id))
            loop = asyncio.get_running_loop()
            clock = loop.time
            loop.time = lambda: clock() + WebAccess._SNAPSHOT_FRAME_BUDGET + 1   # it took the whole budget
            return {"records": []}

        collector_result = {"records": [{"label": "x"}], "crossOriginFrames": [], "context": None}

        async def go():
            with patch.object(WebAccess, '_async_sync_popup_targets', new=AsyncMock()), \
                    patch.object(WebAccess, '_evaluate',
                                 new=AsyncMock(side_effect=[collector_result, "title"])), \
                    patch.object(WebAccess, '_async_list_frame_targets',
                                 new=AsyncMock(return_value=[t1, t2])), \
                    patch.object(WebAccess, '_async_eval_in_frame_target', new=slow_frame_eval), \
                    patch.object(WebAccess, '_render_snapshot_lines', new=Mock(return_value=["[0] x"])), \
                    patch.object(WebAccess, '_format_page_context', new=Mock(return_value="")):
                return await wa._async_take_snapshot()

        result = asyncio.run(go())
        assert visited == ["t1"], "second frame should be skipped after the budget expired"
        assert "Interactive elements" in result
