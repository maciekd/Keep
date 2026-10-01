"""Browser session lifecycle hardening (plan 01): dead-session classification,
health-check/ensure funnel, circuit breaker, browse_close full shutdown, and
child-inclusive Chrome kill. No real Chrome — WebAccess.__new__ + AsyncMock."""

import asyncio
import contextlib
import signal
import threading
import time

import pytest
import websockets.exceptions
from unittest.mock import Mock, patch, AsyncMock

from assistant.tools.WebAccess import WebAccess


def run(coro):
    return asyncio.run(coro)


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
    wa._browse_pdf_text = None
    wa._last_screenshot_bytes = None
    wa._last_screenshot_path = None
    wa._browse_fail_streak = 0
    wa._browser_hard_down = False
    wa._browser_closed_by_user = False
    wa._browser_generation = 1
    wa._browse_session_generation = 1
    for key, value in overrides.items():
        setattr(wa, key, value)
    return wa


@pytest.mark.unit
class TestBrowseOpenClassification:

    def _open_patches(self, snapshot):
        return (
            patch.object(WebAccess, '_async_browse_ready', new=AsyncMock(return_value=None)),
            patch.object(WebAccess, '_async_close_popup_targets', new=AsyncMock()),
            patch.object(WebAccess, '_async_settle_cookie_banner', new=AsyncMock()),
            patch.object(WebAccess, '_async_wait_for_page_ready', new=AsyncMock()),
            patch.object(WebAccess, '_async_take_snapshot', new=snapshot),
        )

    def test_websocket_error_triggers_reinit_and_retry(self):
        wa = make_wa()
        dead_tab = Mock()
        dead_tab.send = AsyncMock(side_effect=websockets.exceptions.WebSocketException(
            "server rejected WebSocket connection: HTTP 500"))
        fresh_tab = Mock()
        fresh_tab.send = AsyncMock()
        wa._tab = dead_tab

        async def fake_init():
            wa._tab = fresh_tab
            wa._mark_browser_up()

        init_mock = AsyncMock(side_effect=fake_init)
        p1, p2, p3, p4, p5 = self._open_patches(AsyncMock(return_value="SNAPSHOT"))
        with p1, p2, p3, p4, p5, \
                patch.object(WebAccess, '_async_initialize_driver', new=init_mock):
            result = run(wa._async_browse_open("https://x.example"))

        assert result == "SNAPSHOT"
        init_mock.assert_awaited_once()
        fresh_tab.send.assert_awaited_once()
        assert wa._browse_session_url == "https://x.example"
        assert wa._browse_session_generation == wa._browser_generation

    def test_page_level_error_probes_but_does_not_reinit_when_alive(self):
        wa = make_wa()
        wa._tab.send = AsyncMock()
        alive = AsyncMock(return_value=True)
        init_mock = AsyncMock()
        p1, p2, p3, p4, p5 = self._open_patches(AsyncMock(side_effect=ValueError("boom")))
        with p1, p2, p3, p4, p5, \
                patch.object(WebAccess, '_async_browser_alive', new=alive), \
                patch.object(WebAccess, '_async_initialize_driver', new=init_mock):
            result = run(wa._async_browse_open("https://x.example"))

        assert result == "ERROR: failed to open the page."
        alive.assert_awaited_once()
        init_mock.assert_not_awaited()

    def test_opaque_error_with_dead_browser_restarts_and_retries_once(self):
        wa = make_wa()
        wa._tab.send = AsyncMock()
        snapshot = AsyncMock(side_effect=[ValueError("transport broke weirdly"), "SNAPSHOT2"])
        ensure = AsyncMock(return_value=True)
        p1, p2, p3, p4, p5 = self._open_patches(snapshot)
        with p1, p2, p3, p4, p5, \
                patch.object(WebAccess, '_async_browser_alive', new=AsyncMock(return_value=False)), \
                patch.object(WebAccess, '_async_ensure_browser', new=ensure):
            result = run(wa._async_browse_open("https://x.example"))

        assert result == "SNAPSHOT2"
        ensure.assert_awaited_once_with(force=True)

    def test_browser_death_after_navigation_reinits_and_feeds_the_breaker(self):
        # The mid-load death path: navigation committed, then Chrome died during
        # readiness/snapshot. Must reinit and register an infra failure — not
        # degrade to the page-level error that skips recovery entirely.
        wa = make_wa()
        wa._tab.send = AsyncMock()
        init_mock = AsyncMock()
        p1, p2, p3, p4, p5 = self._open_patches(
            AsyncMock(side_effect=ConnectionError("browser died mid-load")))
        with p1, p2, p3, p4, p5, \
                patch.object(WebAccess, '_async_initialize_driver', new=init_mock):
            result = run(wa._async_browse_open("https://x.example"))

        assert result == "ERROR: browser crashed while loading the page."
        init_mock.assert_awaited_once()
        assert wa._browse_fail_streak == 1

    def test_opaque_error_still_failing_after_retry_stops_without_second_restart(self):
        wa = make_wa()
        wa._tab.send = AsyncMock()
        snapshot = AsyncMock(side_effect=[ValueError("broke"), ValueError("still broke")])
        ensure = AsyncMock(return_value=True)
        alive = AsyncMock(return_value=False)
        p1, p2, p3, p4, p5 = self._open_patches(snapshot)
        with p1, p2, p3, p4, p5, \
                patch.object(WebAccess, '_async_browser_alive', new=alive), \
                patch.object(WebAccess, '_async_ensure_browser', new=ensure):
            result = run(wa._async_browse_open("https://x.example"))

        assert result == "ERROR: failed to open the page."
        # the _retrying guard: exactly one restart, and the retry pass must not
        # re-probe liveness — that is what prevents unbounded recursion when the
        # browser stays dead
        assert ensure.await_count == 1
        assert alive.await_count == 1


@pytest.mark.unit
class TestBrowserAlive:

    def test_no_browser_is_dead(self):
        wa = make_wa(_browser=None)
        assert run(wa._async_browser_alive()) is False

    def test_probe_success_is_alive(self):
        wa = make_wa()
        wa._browser.connection.send = AsyncMock(return_value=[])
        assert run(wa._async_browser_alive()) is True

    def test_probe_failure_is_dead(self):
        wa = make_wa()
        wa._browser.connection.send = AsyncMock(side_effect=ConnectionError("gone"))
        assert run(wa._async_browser_alive()) is False


@pytest.mark.unit
class TestEnsureBrowser:

    def test_alive_browser_is_a_noop(self):
        wa = make_wa()
        quit_mock = AsyncMock()
        init_mock = AsyncMock()
        with patch.object(WebAccess, '_async_browser_alive', new=AsyncMock(return_value=True)), \
                patch.object(WebAccess, '_async_quit_driver', new=quit_mock), \
                patch.object(WebAccess, '_async_initialize_driver', new=init_mock):
            assert run(wa._async_ensure_browser()) is True
        quit_mock.assert_not_awaited()
        init_mock.assert_not_awaited()

    def test_dead_browser_quits_and_reinits(self):
        wa = make_wa()

        async def fake_quit():
            wa._browser = None
            wa._tab = None

        async def fake_init():
            wa._browser = Mock()
            wa._tab = Mock()

        quit_mock = AsyncMock(side_effect=fake_quit)
        init_mock = AsyncMock(side_effect=fake_init)
        with patch.object(WebAccess, '_async_browser_alive', new=AsyncMock(return_value=False)), \
                patch.object(WebAccess, '_async_quit_driver', new=quit_mock), \
                patch.object(WebAccess, '_async_initialize_driver', new=init_mock):
            assert run(wa._async_ensure_browser()) is True
        quit_mock.assert_awaited_once()
        init_mock.assert_awaited_once()

    def test_backoff_blocks_reinit(self):
        wa = make_wa(_browser=None, _tab=None, _last_init_failure_ts=time.time())
        init_mock = AsyncMock()
        with patch.object(WebAccess, '_async_browser_alive', new=AsyncMock(return_value=False)), \
                patch.object(WebAccess, '_async_initialize_driver', new=init_mock):
            assert run(wa._async_ensure_browser()) is False
        init_mock.assert_not_awaited()

    def test_alive_browser_with_no_tab_restarts_for_a_fresh_tab(self):
        # _adopt_newest_live_target can clear the tab while Chrome stays alive;
        # the health gate must not report healthy with nothing to drive — that
        # bricked every mid-session command until browse_open.
        wa = make_wa(_tab=None)

        async def fake_quit():
            wa._browser = None

        async def fake_init():
            wa._browser = Mock()
            wa._tab = Mock()

        quit_mock = AsyncMock(side_effect=fake_quit)
        init_mock = AsyncMock(side_effect=fake_init)
        with patch.object(WebAccess, '_async_browser_alive', new=AsyncMock(return_value=True)), \
                patch.object(WebAccess, '_async_quit_driver', new=quit_mock), \
                patch.object(WebAccess, '_async_initialize_driver', new=init_mock):
            assert run(wa._async_ensure_browser()) is True
        quit_mock.assert_awaited_once()
        init_mock.assert_awaited_once()

    def test_force_bypasses_backoff(self):
        wa = make_wa(_browser=None, _tab=None, _last_init_failure_ts=time.time())

        async def fake_init():
            wa._browser = Mock()
            wa._tab = Mock()

        init_mock = AsyncMock(side_effect=fake_init)
        with patch.object(WebAccess, '_async_browser_alive', new=AsyncMock(return_value=False)), \
                patch.object(WebAccess, '_async_initialize_driver', new=init_mock):
            assert run(wa._async_ensure_browser(force=True)) is True
        init_mock.assert_awaited_once()


@pytest.mark.unit
class TestMidSessionRecovery:

    def test_act_after_restart_reopens_and_returns_note(self):
        wa = make_wa(_browse_session_url="https://shop.example",
                     _browse_session_generation=1, _browser_generation=2,
                     _browse_snapshot_frame_ids=[None])
        reopen = AsyncMock(return_value="FRESH SNAPSHOT")
        act = AsyncMock()
        with patch.object(WebAccess, '_async_ensure_browser', new=AsyncMock(return_value=True)), \
                patch.object(WebAccess, '_async_browse_open', new=reopen), \
                patch.object(WebAccess, '_async_execute_action', new=act):
            result = run(wa._async_browse_act("0"))

        assert result.startswith("NOTE: the browser was restarted")
        assert "FRESH SNAPSHOT" in result
        reopen.assert_awaited_once_with("https://shop.example")
        act.assert_not_awaited()

    def test_gate_is_noop_when_no_restart_happened(self):
        wa = make_wa(_browse_session_url="https://shop.example")
        reopen = AsyncMock()
        with patch.object(WebAccess, '_async_ensure_browser', new=AsyncMock(return_value=True)), \
                patch.object(WebAccess, '_async_browse_open', new=reopen):
            assert run(wa._async_browse_ready_midsession()) is None
        reopen.assert_not_awaited()

    def test_failed_reopen_returns_the_error_without_note(self):
        wa = make_wa(_browse_session_url="https://shop.example",
                     _browse_session_generation=1, _browser_generation=2)
        reopen = AsyncMock(return_value="ERROR: failed to open the page.")
        with patch.object(WebAccess, '_async_ensure_browser', new=AsyncMock(return_value=True)), \
                patch.object(WebAccess, '_async_browse_open', new=reopen):
            result = run(wa._async_browse_ready_midsession())
        assert result == "ERROR: failed to open the page."


@pytest.mark.unit
class TestCircuitBreaker:

    def test_three_failures_force_one_restart_then_latch_hard_down(self):
        wa = make_wa(_browser=None, _tab=None)
        ensure = AsyncMock(return_value=False)
        with patch.object(WebAccess, '_async_ensure_browser', new=ensure):
            r1 = run(wa._async_browse_ready())
            r2 = run(wa._async_browse_ready())
            r3 = run(wa._async_browse_ready())
            r4 = run(wa._async_browse_ready())

        assert r1 == "ERROR: browser is not available on this server."
        assert r2 == r1
        assert r3 == WebAccess._HARD_DOWN_ERROR
        assert r4 == WebAccess._HARD_DOWN_ERROR
        assert wa._browser_hard_down is True
        # 3 normal attempts + 1 forced; the latched 4th command never probes
        assert ensure.await_count == 4
        forced = [c for c in ensure.await_args_list if c.kwargs.get('force')]
        assert len(forced) == 1

    def test_forced_restart_success_resets_streak_and_continues(self):
        wa = make_wa(_browser=None, _tab=None, _browse_fail_streak=2)

        async def ensure_side(force=False):
            if force:
                wa._tab = Mock()
                return True
            return False

        with patch.object(WebAccess, '_async_ensure_browser', new=AsyncMock(side_effect=ensure_side)):
            result = run(wa._async_browse_ready())

        assert result is None
        assert wa._browse_fail_streak == 0
        assert wa._browser_hard_down is False

    def test_successful_open_resets_accumulated_fail_streak(self):
        # Two transient infra failures early in a session must not linger and
        # combine with a later unrelated one to trip the breaker: a successful
        # open proves the browser works and clears the streak.
        wa = make_wa(_browse_fail_streak=2)
        wa._tab.send = AsyncMock()
        with patch.object(WebAccess, '_async_browse_ready', new=AsyncMock(return_value=None)), \
                patch.object(WebAccess, '_async_close_popup_targets', new=AsyncMock()), \
                patch.object(WebAccess, '_async_settle_cookie_banner', new=AsyncMock()), \
                patch.object(WebAccess, '_async_wait_for_page_ready', new=AsyncMock()), \
                patch.object(WebAccess, '_async_take_snapshot', new=AsyncMock(return_value="SNAP")):
            result = run(wa._async_browse_open("https://x.example"))

        assert result == "SNAP"
        assert wa._browse_fail_streak == 0

    def test_page_level_error_does_not_count_as_infra_failure(self):
        wa = make_wa(_browse_session_url="https://shop.example",
                     _browse_snapshot_frame_ids=[None])
        with patch.object(WebAccess, '_async_ensure_browser', new=AsyncMock(return_value=True)), \
                patch.object(WebAccess, '_async_take_snapshot', new=AsyncMock(return_value="SNAP")):
            result = run(wa._async_browse_act("5"))
        assert "out of range" in result
        assert wa._browse_fail_streak == 0

    def test_mark_browser_up_clears_breaker_and_latch(self):
        wa = make_wa(_browse_fail_streak=5, _browser_hard_down=True,
                     _browser_closed_by_user=True, _browser_generation=3)
        wa._mark_browser_up()
        assert wa._browse_fail_streak == 0
        assert wa._browser_hard_down is False
        assert wa._browser_closed_by_user is False
        assert wa._browser_generation == 4


@pytest.mark.unit
class TestBrowseCloseFullShutdown:

    def test_close_kills_chrome_and_resets_everything(self):
        wa = make_wa(_browse_session_url="https://x.example", _browse_pdf_text="pdf",
                     _browse_fail_streak=2, _browser_hard_down=True)
        browser = Mock(spec=['stop', '_process', '_process_pid'])
        proc = Mock()
        proc.pid = 555
        browser._process = proc
        browser._process_pid = 555
        wa._browser = browser
        child = Mock()
        child.pid = 556

        with patch('assistant.tools.WebAccess.os.kill') as mock_kill, \
                patch.object(WebAccess, '_chrome_child_processes',
                             staticmethod(lambda pid: [child] if pid == 555 else [])), \
                patch.object(WebAccess, '_async_wait_for_chrome_exit', new=AsyncMock()):
            result = run(wa._async_browse_close())

        assert result == "Browse session closed."
        browser.stop.assert_called_once()
        child.kill.assert_called_once()
        mock_kill.assert_called_once_with(555, signal.SIGKILL)
        assert wa._browser is None
        assert wa._tab is None
        assert wa._browse_session_url is None
        assert wa._browse_pdf_text is None
        assert wa._browse_fail_streak == 0
        assert wa._browser_hard_down is False
        assert wa._browser_closed_by_user is True

    def test_next_browse_open_relaunches_lazily(self):
        wa = make_wa(_browser=None, _tab=None, _browser_closed_by_user=True)
        fresh_tab = Mock()
        fresh_tab.send = AsyncMock()

        async def fake_init():
            wa._browser = Mock()
            wa._tab = fresh_tab
            wa._mark_browser_up()

        init_mock = AsyncMock(side_effect=fake_init)
        with patch.object(WebAccess, '_async_initialize_driver', new=init_mock), \
                patch.object(WebAccess, '_async_close_popup_targets', new=AsyncMock()), \
                patch.object(WebAccess, '_async_settle_cookie_banner', new=AsyncMock()), \
                patch.object(WebAccess, '_async_wait_for_page_ready', new=AsyncMock()), \
                patch.object(WebAccess, '_async_take_snapshot', new=AsyncMock(return_value="SNAP")):
            result = run(wa._async_browse_open("https://y.example"))

        assert result == "SNAP"
        init_mock.assert_awaited_once()
        assert wa._browse_session_url == "https://y.example"
        assert wa._browser_closed_by_user is False

    def test_check_health_tolerates_deliberate_close(self):
        wa = make_wa(_browser=None, _browser_closed_by_user=True)
        assert wa.check_health() is True
        wa._browser_closed_by_user = False
        assert wa.check_health() is False

    def test_close_resets_init_backoff_so_reopen_gets_a_real_attempt(self):
        # The documented recovery path is close → open; when the previous init
        # failed, browse_close must clear the failure timestamp or the reopen
        # is swallowed by the init backoff and fails again.
        wa = make_wa(_last_init_failure_ts=time.time())
        with patch.object(WebAccess, '_async_quit_driver', new=AsyncMock()):
            result = run(wa._async_browse_close())
        assert result == "Browse session closed."
        assert wa._last_init_failure_ts == 0.0

    def test_sync_fallback_close_also_resets_init_backoff(self):
        wa = make_wa(_last_init_failure_ts=time.time())
        wa._loop_thread = Mock()
        wa._loop_thread.submit = Mock(side_effect=RuntimeError("loop thread dead"))
        with patch.object(WebAccess, '_async_browse_close', new=Mock(return_value=None)), \
                patch.object(WebAccess, '_kill_tracked_chrome_pids', new=Mock()):
            result = wa.browse_close()
        assert result == "Browse session closed."
        assert wa._last_init_failure_ts == 0.0


@pytest.mark.unit
class TestCdpWaitFor:
    """Soft CDP timeouts must not cancel nodriver's in-flight Transaction: a
    cancelled Transaction stays in the connection mapper, so the late reply
    set_result()s a cancelled future inside the websocket listener task, which
    kills the listener and hangs every later CDP call on that connection."""

    def test_result_passes_through_under_the_cap(self):
        async def scenario():
            async def send():
                return "reply"
            assert await WebAccess._cdp_wait_for(send(), 1.0) == "reply"
        run(scenario())

    def test_exception_passes_through_under_the_cap(self):
        async def scenario():
            async def send():
                raise ConnectionError("dead transport")
            with pytest.raises(ConnectionError):
                await WebAccess._cdp_wait_for(send(), 1.0)
        run(scenario())

    class LateReply:
        """A CDP call that answers only when the test lets it — after the caller's timeout —
        and records how it ended."""

        def __init__(self):
            self.cancelled = self.finished = False
            self._release = asyncio.Event()
            self._settled = asyncio.Event()

        async def call(self, *_args):
            try:
                await self._release.wait()
                self.finished = True
                return "late-reply"
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            finally:
                self._settled.set()

        async def arrive(self):
            self._release.set()
            await asyncio.wait_for(self._settled.wait(), 5)

    def test_timeout_abandons_the_call_without_cancelling_it(self):
        reply = self.LateReply()

        async def scenario():
            with pytest.raises(asyncio.TimeoutError):
                await WebAccess._cdp_wait_for(reply.call(), 0.01)
            await reply.arrive()

        run(scenario())
        assert not reply.cancelled
        assert reply.finished

    def test_late_failure_of_abandoned_call_is_swallowed(self):
        # The abandoned task's eventual error must be retrieved, or asyncio
        # reports 'Task exception was never retrieved' via the loop handler.
        import gc
        reported = []

        async def scenario():
            asyncio.get_running_loop().set_exception_handler(
                lambda loop, ctx: reported.append(ctx))

            async def failing_send():
                await asyncio.sleep(0.02)
                raise RuntimeError("late transport error")
            with pytest.raises(asyncio.TimeoutError):
                await WebAccess._cdp_wait_for(failing_send(), 0.005)
            await asyncio.sleep(0.05)
            gc.collect()

        run(scenario())
        assert reported == []

    def test_slow_navigate_times_out_softly_without_cancelling_the_send(self):
        wa = make_wa()
        reply = self.LateReply()
        tab = Mock()
        tab.send = reply.call

        async def scenario():
            with pytest.raises(asyncio.TimeoutError):
                await wa._async_navigate(tab, "https://x.example")
            await reply.arrive()

        with patch.object(WebAccess, '_NAV_COMMIT_TIMEOUT', 0.01):
            run(scenario())
        assert reply.finished and not reply.cancelled

    def test_slow_readiness_eval_times_out_softly_without_cancelling_it(self):
        wa = make_wa()
        reply = self.LateReply()

        async def scenario():
            status = await wa._async_wait_for_page_ready(Mock())
            assert status == "soft-timeout:evaluate"
            await reply.arrive()

        with patch.object(WebAccess, '_evaluate', new=lambda self, tab, expr: reply.call()), \
                patch.object(WebAccess, '_PAGE_READY_EVAL_TIMEOUT', 0.01):
            run(scenario())
        assert reply.finished and not reply.cancelled


@pytest.mark.unit
class TestQuitDriverChildKill:

    def test_quit_driver_kills_helper_children(self):
        wa = make_wa()
        browser = Mock(spec=['stop', '_process', '_process_pid'])
        proc = Mock()
        proc.pid = 300
        browser._process = proc
        browser._process_pid = 300
        wa._browser = browser
        child = Mock()
        child.pid = 301

        with patch('assistant.tools.WebAccess.os.kill') as mock_kill, \
                patch.object(WebAccess, '_chrome_child_processes', staticmethod(lambda pid: [child])), \
                patch.object(WebAccess, '_async_wait_for_chrome_exit', new=AsyncMock()):
            run(wa._async_quit_driver())

        browser.stop.assert_called_once()
        child.kill.assert_called_once()
        mock_kill.assert_called_once_with(300, signal.SIGKILL)
        assert wa._browser is None
        assert wa._tab is None


@pytest.mark.unit
class TestQuitDriverWaitsForExit:
    """browse_close immediately followed by browse_open used to launch into the
    still-dying Chrome tree, fail with 'Failed to connect to browser' and wipe
    the profile. Quit must wait for the tree it killed — collected before the
    kill, since killing clears the tracked PIDs."""

    def test_waits_on_the_tree_collected_before_the_kill(self):
        wa = make_wa()
        browser = Mock(spec=['stop', '_process', '_process_pid'])
        proc = Mock()
        proc.pid = 300
        browser._process = proc
        browser._process_pid = 300
        wa._browser = browser
        parent, child = Mock(pid=300), Mock(pid=301)
        order = []

        def fake_kill(*_a, **_k):
            order.append('kill')

        def fake_wait(procs, timeout):
            order.append(('wait', list(procs), timeout))
            return [], []

        with patch('assistant.tools.WebAccess.os.kill', side_effect=fake_kill), \
                patch('assistant.tools.WebAccess.psutil.Process', return_value=parent), \
                patch('assistant.tools.WebAccess.psutil.wait_procs', side_effect=fake_wait), \
                patch.object(WebAccess, '_chrome_child_processes', staticmethod(lambda pid: [child])):
            run(wa._async_quit_driver())

        assert order[0] == 'kill'
        assert order[-1] == ('wait', [parent, child], WebAccess._CHROME_EXIT_TIMEOUT)
        assert wa._chrome_pids == set()

    def test_survivors_do_not_fail_the_quit(self):
        wa = make_wa()
        wa._browser = Mock(spec=['stop', '_process', '_process_pid'], _process=Mock(pid=300),
                           _process_pid=300)
        survivor = Mock(pid=300)
        with patch('assistant.tools.WebAccess.os.kill'), \
                patch('assistant.tools.WebAccess.psutil.Process', return_value=survivor), \
                patch('assistant.tools.WebAccess.psutil.wait_procs', return_value=([], [survivor])), \
                patch.object(WebAccess, '_chrome_child_processes', staticmethod(lambda pid: [])):
            run(wa._async_quit_driver())
        assert wa._browser is None


@pytest.mark.unit
class TestInitializeDriverContract:
    """The post-start contract of the real _async_initialize_driver. Every other
    lifecycle test replaces the whole method with an AsyncMock and hand-rolls its
    effects, so the generation bump / breaker reset / known-target seeding the
    method must perform were unpinned — deleting them broke nothing."""

    def _init_patches(self, start_mock, page_targets):
        return (
            patch.object(WebAccess, '_async_quit_driver', new=AsyncMock()),
            patch.object(WebAccess, '_find_chrome_binary', return_value='/fake/chrome'),
            patch.object(WebAccess, '_kill_orphan_chrome_using_profile'),
            patch.object(WebAccess, '_clear_singleton_locks'),
            patch.object(WebAccess, '_track_chrome_pids'),
            patch.object(WebAccess, '_launch_user_agent', return_value='Mozilla/5.0 test Chrome/154.0.0.0'),
            patch.object(WebAccess, '_async_install_stealth_script', new=AsyncMock()),
            patch.object(WebAccess, '_async_setup_network_monitor', new=AsyncMock()),
            patch.object(WebAccess, '_page_targets', return_value=page_targets),
            patch('assistant.tools.WebAccess.nodriver.start', new=start_mock),
        )

    @contextlib.contextmanager
    def _patched(self, start_mock, page_targets):
        with contextlib.ExitStack() as stack:
            for p in self._init_patches(start_mock, page_targets):
                stack.enter_context(p)
            yield

    def test_successful_init_marks_up_and_seeds_known_targets(self):
        wa = make_wa(_browser=None, _tab=None, _profile_dir='/tmp/floxy-test-profile',
                     _pending_popup_targets={'stale': 1.0}, _browser_generation=3,
                     _browse_fail_streak=2, _browser_hard_down=True,
                     _browser_closed_by_user=True, _last_init_failure_ts=123.0)
        base_tab = Mock()
        base_tab.send = AsyncMock()
        base_tab.target.target_id = 'base'
        leftover = Mock()
        leftover.target.target_id = 'leftover'
        browser = Mock()
        browser.get = AsyncMock(return_value=base_tab)
        start_mock = AsyncMock(return_value=browser)
        with self._patched(start_mock, [leftover]):
            run(wa._async_initialize_driver())

        assert wa._browser is browser
        assert wa._tab is base_tab
        # The UA goes on Chrome's command line (so Chrome derives real client hints
        # and popups inherit it) — never through Emulation.setUserAgentOverride,
        # which DataDome hard-blocks on. And no --disable-gpu: SwiftShader is a tell.
        args = start_mock.await_args.kwargs['browser_args']
        assert '--user-agent=Mozilla/5.0 test Chrome/154.0.0.0' in args
        assert '--disable-gpu' not in args
        assert not any('setUserAgentOverride' in str(c) or 'setExtraHTTPHeaders' in str(c)
                       for c in base_tab.send.await_args_list)
        # _mark_browser_up ran: mid-session restart detection needs the bump,
        # and the breaker/latch state must clear on a successful launch
        assert wa._browser_generation == 4
        assert wa._browse_fail_streak == 0
        assert wa._browser_hard_down is False
        assert wa._browser_closed_by_user is False
        assert wa._last_init_failure_ts == 0.0
        # seeding: base tab + every pre-existing page target, so the first popup
        # sync after (re)init cannot adopt a leftover target as a fresh popup
        assert wa._known_page_targets == {'base', 'leftover'}
        assert wa._pending_popup_targets == {}

    def test_post_start_failure_stops_browser_and_stamps_backoff(self):
        wa = make_wa(_browser=None, _tab=None, _profile_dir='/tmp/floxy-test-profile',
                     _pending_popup_targets={})
        browser = Mock()
        browser.get = AsyncMock(side_effect=ConnectionError("tab open failed"))
        with self._patched(AsyncMock(return_value=browser), []):
            run(wa._async_initialize_driver())

        browser.stop.assert_called_once()
        assert wa._browser is None
        assert wa._tab is None
        assert wa._last_init_failure_ts > 0

    def _fresh(self):
        return make_wa(_browser=None, _tab=None, _profile_dir='/tmp/floxy-test-profile',
                       _pending_popup_targets={}, _INIT_INTER_RETRY_DELAY=0)

    def _browser(self):
        tab = Mock()
        tab.send = AsyncMock()
        tab.target.target_id = 'base'
        browser = Mock()
        browser.get = AsyncMock(return_value=tab)
        return browser

    def test_first_start_failure_retries_without_wiping_the_profile(self):
        # The profile holds every login; a start that collides with the previous
        # Chrome tree still dying (browse_close → browse_open) must not cost it.
        wa = self._fresh()
        start_mock = AsyncMock(side_effect=[RuntimeError('Failed to connect to browser'), self._browser()])
        with self._patched(start_mock, []), patch.object(WebAccess, '_wipe_profile') as wipe:
            run(wa._async_initialize_driver())
        assert wa._browser is not None
        assert start_mock.await_count == 2
        wipe.assert_not_called()

    def test_profile_is_wiped_only_for_the_last_attempt(self):
        wa = self._fresh()
        start_mock = AsyncMock(side_effect=[RuntimeError('1'), RuntimeError('2'), self._browser()])
        with self._patched(start_mock, []), patch.object(WebAccess, '_wipe_profile') as wipe:
            run(wa._async_initialize_driver())
        assert wa._browser is not None
        assert start_mock.await_count == 3
        wipe.assert_called_once()

    def test_three_failures_stamp_the_backoff(self):
        wa = self._fresh()
        start_mock = AsyncMock(side_effect=RuntimeError('down'))
        with self._patched(start_mock, []), patch.object(WebAccess, '_wipe_profile'):
            run(wa._async_initialize_driver())
        assert wa._browser is None
        assert start_mock.await_count == 3
        assert wa._last_init_failure_ts > 0


@pytest.mark.unit
class TestFetchViaChrome:
    """_async_fetch_via_chrome (the fetch_webpage tier-1 path): ensure-browser
    gate, readiness-before-consent ordering, dead-session retry, and
    opaque-failure recovery. Only the sync wrapper was covered before."""

    def _fetch_patches(self, order=None):
        def rec(name):
            async def _f(self, *args, **kwargs):
                if order is not None:
                    order.append(name)
            return _f
        return (
            patch.object(WebAccess, '_async_wait_for_page_ready', new=rec('ready')),
            patch.object(WebAccess, '_async_settle_cookie_banner', new=rec('cookies')),
            patch.object(WebAccess, '_async_scroll_to_load_content', new=rec('scroll')),
        )

    def test_ensure_browser_gate_blocks_fetch(self):
        wa = make_wa()
        wa._tab.get = AsyncMock()
        with patch.object(WebAccess, '_async_ensure_browser', new=AsyncMock(return_value=False)):
            assert run(wa._async_fetch_via_chrome("https://x.example")) is None
        wa._tab.get.assert_not_awaited()

    def test_readiness_runs_before_cookie_settle_before_scroll(self):
        wa = make_wa()
        order = []
        wa._tab.get = AsyncMock(side_effect=lambda url: order.append('get'))
        wa._tab.get_content = AsyncMock(return_value='<html>x</html>')
        p1, p2, p3 = self._fetch_patches(order)
        with p1, p2, p3, \
                patch.object(WebAccess, '_async_ensure_browser', new=AsyncMock(return_value=True)):
            result = run(wa._async_fetch_via_chrome("https://x.example"))
        assert result == '<html>x</html>'
        assert order == ['get', 'ready', 'cookies', 'scroll']

    def test_dead_session_on_get_reinits_and_retries(self):
        wa = make_wa()
        dead_tab = Mock()
        dead_tab.get = AsyncMock(side_effect=websockets.exceptions.WebSocketException(
            "server rejected WebSocket connection: HTTP 500"))
        fresh_tab = Mock()
        fresh_tab.get = AsyncMock()
        fresh_tab.get_content = AsyncMock(return_value='CONTENT')
        wa._tab = dead_tab

        async def fake_init():
            wa._tab = fresh_tab

        init_mock = AsyncMock(side_effect=fake_init)
        p1, p2, p3 = self._fetch_patches()
        with p1, p2, p3, \
                patch.object(WebAccess, '_async_ensure_browser', new=AsyncMock(return_value=True)), \
                patch.object(WebAccess, '_async_initialize_driver', new=init_mock):
            result = run(wa._async_fetch_via_chrome("https://x.example"))
        assert result == 'CONTENT'
        init_mock.assert_awaited_once()
        fresh_tab.get.assert_awaited_once()

    def test_dead_session_with_failed_reinit_returns_none(self):
        wa = make_wa()
        wa._tab.get = AsyncMock(side_effect=websockets.exceptions.WebSocketException("HTTP 500"))

        async def fake_init():
            wa._tab = None

        p1, p2, p3 = self._fetch_patches()
        with p1, p2, p3, \
                patch.object(WebAccess, '_async_ensure_browser', new=AsyncMock(return_value=True)), \
                patch.object(WebAccess, '_async_initialize_driver', new=AsyncMock(side_effect=fake_init)):
            assert run(wa._async_fetch_via_chrome("https://x.example")) is None

    def test_opaque_failure_with_dead_browser_reinitializes(self):
        wa = make_wa()
        wa._tab.get = AsyncMock()
        wa._tab.get_content = AsyncMock(side_effect=ValueError("boom"))
        init_mock = AsyncMock()
        p1, p2, p3 = self._fetch_patches()
        with p1, p2, p3, \
                patch.object(WebAccess, '_async_ensure_browser', new=AsyncMock(return_value=True)), \
                patch.object(WebAccess, '_async_browser_alive', new=AsyncMock(return_value=False)), \
                patch.object(WebAccess, '_async_initialize_driver', new=init_mock):
            assert run(wa._async_fetch_via_chrome("https://x.example")) is None
        init_mock.assert_awaited_once()

    def test_opaque_failure_with_live_browser_does_not_reinit(self):
        wa = make_wa()
        wa._tab.get = AsyncMock()
        wa._tab.get_content = AsyncMock(side_effect=ValueError("boom"))
        init_mock = AsyncMock()
        p1, p2, p3 = self._fetch_patches()
        with p1, p2, p3, \
                patch.object(WebAccess, '_async_ensure_browser', new=AsyncMock(return_value=True)), \
                patch.object(WebAccess, '_async_browser_alive', new=AsyncMock(return_value=True)), \
                patch.object(WebAccess, '_async_initialize_driver', new=init_mock):
            assert run(wa._async_fetch_via_chrome("https://x.example")) is None
        init_mock.assert_not_awaited()


@pytest.mark.unit
class TestBrowseOpenRestartFailure:
    """A reinit funnel that fails to produce a tab must register an infra failure
    (feeding the circuit breaker), not crash or pretend the page opened."""

    def _open_patches(self):
        return (
            patch.object(WebAccess, '_async_browse_ready', new=AsyncMock(return_value=None)),
            patch.object(WebAccess, '_async_close_popup_targets', new=AsyncMock()),
        )

    def test_no_tab_after_popup_cleanup_and_failed_reinit_is_infra_error(self):
        wa = make_wa(_tab=None)
        init_mock = AsyncMock()  # leaves _tab None: the relaunch failed
        p1, p2 = self._open_patches()
        with p1, p2, patch.object(WebAccess, '_async_initialize_driver', new=init_mock):
            result = run(wa._async_browse_open("https://x.example"))
        assert result == "ERROR: browser could not be restarted."
        init_mock.assert_awaited_once()
        assert wa._browse_fail_streak == 1

    def test_dead_nav_with_failed_reinit_is_infra_error(self):
        wa = make_wa()
        wa._tab.send = AsyncMock(side_effect=websockets.exceptions.WebSocketException(
            "server rejected WebSocket connection: HTTP 500"))

        async def fake_init():
            wa._tab = None

        p1, p2 = self._open_patches()
        with p1, p2, patch.object(WebAccess, '_async_initialize_driver',
                                  new=AsyncMock(side_effect=fake_init)):
            result = run(wa._async_browse_open("https://x.example"))
        assert result == "ERROR: browser could not be restarted."
        assert wa._browse_fail_streak == 1


@pytest.mark.unit
class TestBrowseOpenPdfSession:

    def test_pdf_session_stamps_browser_generation(self):
        # Without the stamp, a PDF session opened at generation N would make
        # _async_browse_ready_midsession think the browser restarted.
        wa = make_wa(_browser_generation=7, _browse_session_generation=0)
        with patch.object(WebAccess, '_async_browse_ready', new=AsyncMock(return_value=None)), \
                patch.object(WebAccess, '_async_close_popup_targets', new=AsyncMock()), \
                patch.object(WebAccess, '_async_fetch_pdf_bytes_via_chrome',
                             new=AsyncMock(return_value=b'%PDF-1.4')), \
                patch('assistant.tools.WebAccess.PdfExtractor.extract_text', return_value='PDF TEXT'):
            result = run(wa._async_browse_open("https://x.example/manual.pdf"))
        assert result.startswith("PDF document loaded")
        assert wa._browse_pdf_text == 'PDF TEXT'
        assert wa._browse_session_url == "https://x.example/manual.pdf"
        assert wa._browse_session_generation == 7


@pytest.mark.unit
class TestBrowseCloseLoopThreadFallback:

    def test_loop_thread_failure_still_kills_chrome_and_resets(self):
        # A dead/hung loop thread is exactly the state browse_close exists to
        # recover from — the sync fallback must still kill Chrome and reset state.
        wa = make_wa(_browse_session_url="https://x.example", _browse_pdf_text="pdf",
                     _browse_fail_streak=2, _browser_hard_down=True)
        wa._loop_thread = Mock()
        wa._loop_thread.submit = Mock(side_effect=RuntimeError("loop thread dead"))
        kill_mock = Mock()
        with patch.object(WebAccess, '_async_browse_close', new=Mock(return_value=None)), \
                patch.object(WebAccess, '_kill_tracked_chrome_pids', new=kill_mock):
            result = wa.browse_close()
        assert result == "Browse session closed."
        wa._loop_thread.submit.assert_called_once()
        kill_mock.assert_called_once()
        assert wa._browser is None
        assert wa._tab is None
        assert wa._browse_session_url is None
        assert wa._browse_pdf_text is None
        assert wa._browse_fail_streak == 0
        assert wa._browser_hard_down is False
        assert wa._browser_closed_by_user is True
