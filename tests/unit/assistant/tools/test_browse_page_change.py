"""After a click/Enter the snapshot must show the page the action produced, not the one
the LLM already saw: the page is watched through its text signature — the change-wait
before the readiness gate, the settle wait, the one-retry on an unchanged page, and the
NOTE when the page really did not move."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from assistant.tools.WebAccess import WebAccess
from tests.unit.assistant.tools.virtual_time import virtual_time


def make_wa(url='https://shop.example/kassa'):
    wa = WebAccess.__new__(WebAccess)
    wa._browser = None
    wa._tab = SimpleNamespace(send=AsyncMock(), target=SimpleNamespace(url=url))
    wa._browse_session_url = url
    wa._browse_pdf_text = None
    wa._browse_frame_sessions = {}
    wa._browse_snapshot_frame_ids = [None] * 8
    wa._browse_snapshot_labels = []
    wa._pending_target_notes = []
    wa._dialog_notes = []
    return wa


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def fast_waits(monkeypatch):
    """The caps and polls count the virtual clock's seconds: on the real one, a machine that
    stalled 40 ms between two probes had used the settle cap up before the page was steady."""
    monkeypatch.setattr(WebAccess, "_POPUP_SETTLE_S", 0)
    monkeypatch.setattr(WebAccess, "_ACTION_CHANGE_TIMEOUT", 0.05)
    monkeypatch.setattr(WebAccess, "_ACTION_CHANGE_POLL", 0.005)
    monkeypatch.setattr(WebAccess, "_ACTION_UNCHANGED_RETRY_S", 0)
    monkeypatch.setattr(WebAccess, "_SETTLE_TIMEOUT", 0.05)
    monkeypatch.setattr(WebAccess, "_SETTLE_POLL", 0.005)
    with virtual_time() as clock:
        yield clock


def probe(state='complete', sig='u#10:1', busy=False):
    return {'state': state, 'sig': sig, 'busy': busy}


def probes(*seq):
    """_evaluate answering the page probe with each dict in turn, the last one repeating.
    (_evaluate is a staticmethod, so the stand-in must take whatever it is called with.)"""
    seq = list(seq)

    async def ev(*args, **kw):
        return seq.pop(0) if len(seq) > 1 else seq[0]
    return patch.object(WebAccess, '_evaluate', new=ev)


@pytest.mark.unit
class TestActionMark:

    def test_records_the_moment_of_the_action(self, fast_waits):
        wa = make_wa()
        with probes(probe()):
            mark = run(wa._async_action_mark())
        assert mark['net_ts'] == fast_waits.now

    def test_records_the_text_signature(self):
        wa = make_wa()
        with probes(probe(sig='S1')):
            assert run(wa._async_action_mark())['sig'] == 'S1'

    def test_probe_failure_leaves_no_signature(self):
        wa = make_wa()
        with patch.object(WebAccess, '_evaluate', new=AsyncMock(side_effect=asyncio.TimeoutError())):
            assert run(wa._async_action_mark())['sig'] is None

    def test_non_dict_probe_leaves_no_signature(self):
        wa = make_wa()
        with patch.object(WebAccess, '_evaluate', new=AsyncMock(return_value='iframe')):
            assert run(wa._async_action_mark())['sig'] is None


@pytest.mark.unit
class TestWaitForPageChange:

    def test_text_change_is_seen(self):
        wa = make_wa()
        with probes(probe(sig='S1'), probe(sig='S2')):
            status, last = run(wa._async_wait_for_page_change({'sig': 'S1'}))
        assert status == 'changed'
        assert last['sig'] == 'S2'

    def test_navigation_in_progress(self):
        wa = make_wa()
        with probes(probe(state='loading', sig='S1')):
            status, _ = run(wa._async_wait_for_page_change({'sig': 'S1'}))
        assert status == 'navigating'

    def test_quiet_page_times_out_softly(self):
        wa = make_wa()
        with probes(probe(sig='S1')):
            status, last = run(wa._async_wait_for_page_change({'sig': 'S1'}))
        assert status == 'timeout'
        assert last['sig'] == 'S1'

    def test_unknown_mark_counts_as_changed(self):
        wa = make_wa()
        with probes(probe(sig='S1')):
            status, _ = run(wa._async_wait_for_page_change({'sig': None}))
        assert status == 'changed'

    def test_probe_failure_is_soft(self):
        wa = make_wa()
        with patch.object(WebAccess, '_evaluate', new=AsyncMock(side_effect=RuntimeError('gone'))):
            assert run(wa._async_wait_for_page_change({'sig': 'S1'})) == ('probe-failed', None)


@pytest.mark.unit
class TestWaitForSettle:

    def test_steady_page_costs_one_probe(self):
        wa = make_wa()
        calls = []

        async def ev(*args, **kw):
            calls.append(1)
            return probe(sig='S1')
        with patch.object(WebAccess, '_evaluate', new=ev):
            assert run(wa._async_wait_for_settle(wa._tab, 'S1')) == 'settled'
        assert len(calls) == 1

    def test_without_a_known_signature_two_equal_probes_are_needed(self):
        wa = make_wa()
        calls = []

        async def ev(*args, **kw):
            calls.append(1)
            return probe(sig='S1')
        with patch.object(WebAccess, '_evaluate', new=ev):
            assert run(wa._async_wait_for_settle(wa._tab)) == 'settled'
        assert len(calls) == 2

    def test_changing_text_is_waited_out(self):
        wa = make_wa()
        with probes(probe(sig='S1'), probe(sig='S2'), probe(sig='S3'), probe(sig='S3')):
            assert run(wa._async_wait_for_settle(wa._tab, 'S1')) == 'settled'

    def test_progress_indicator_holds_the_wait_until_it_goes(self):
        wa = make_wa()
        with probes(probe(sig='S1', busy=True), probe(sig='S1', busy=True), probe(sig='S1')):
            assert run(wa._async_wait_for_settle(wa._tab, 'S1')) == 'settled'

    def test_spinner_that_never_stops_hits_the_soft_cap(self):
        wa = make_wa()
        with probes(probe(sig='S1', busy=True)):
            assert run(wa._async_wait_for_settle(wa._tab, 'S1')) == 'timeout'

    def test_probe_failure_is_soft(self):
        wa = make_wa()
        with patch.object(WebAccess, '_evaluate', new=AsyncMock(return_value=None)):
            assert run(wa._async_wait_for_settle(wa._tab, 'S1')) == 'probe-failed'


def tail_patches(snapshots, sigs, change=('changed', None)):
    """_async_take_snapshot yields the given snapshots in turn; the page probe answers
    with the given signatures in turn (the last repeating)."""
    sigs = list(sigs)

    async def take(self, requested_url=None):
        return snapshots.pop(0)

    async def ev(*args, **kw):
        return probe(sig=sigs.pop(0) if len(sigs) > 1 else sigs[0])
    return (
        patch.object(WebAccess, '_async_wait_for_page_change', new=AsyncMock(return_value=change)),
        patch.object(WebAccess, '_async_sync_popup_targets', new=AsyncMock()),
        patch.object(WebAccess, '_async_wait_for_page_ready', new=AsyncMock()),
        patch.object(WebAccess, '_async_wait_for_settle', new=AsyncMock(return_value='settled')),
        patch.object(WebAccess, '_async_take_snapshot', new=take),
        patch.object(WebAccess, '_evaluate', new=ev),
    )


@pytest.mark.unit
class TestSnapshotAfterAction:

    def test_changed_page_returns_the_snapshot_at_once(self):
        wa = make_wa()
        mark = {'sig': 'OLD'}
        p = tail_patches(['NEW'], ['NEW'])
        with p[0] as change, p[1], p[2], p[3] as settle, p[4], p[5]:
            assert run(wa._async_snapshot_after_action(mark, "browse_act('3')")) == 'NEW'
        change.assert_awaited_once_with(mark)
        settle.assert_awaited_once()

    def test_unchanged_page_is_retried_once_and_the_retry_wins(self):
        wa = make_wa()
        mark = {'sig': 'OLD'}
        p = tail_patches(['S1', 'S2'], ['OLD', 'NEW'])
        with p[0], p[1], p[2], p[3], p[4], p[5]:
            assert run(wa._async_snapshot_after_action(mark, "browse_act('3')")) == 'S2'

    def test_still_unchanged_after_the_retry_is_flagged(self):
        wa = make_wa()
        mark = {'sig': 'OLD'}
        p = tail_patches(['S1', 'S2'], ['OLD'])
        with p[0], p[1], p[2], p[3], p[4], p[5]:
            result = run(wa._async_snapshot_after_action(mark, "browse_act('3')"))
        assert result.startswith("NOTE: the page looks exactly as it did before browse_act('3')")
        assert result.endswith("\n\nS2")
        assert 'browse_snapshot()' in result

    def test_the_flag_lists_the_requests_the_action_sent_that_the_server_refused(self):
        # The 2026-10-22 Tesla checkout: Nästa drew nothing, the 400 behind it sat in the
        # capture — the NOTE must carry it, response body included, not leave it to a
        # browse_network call rounds later.
        wa = make_wa()
        wa._clear_network_log()
        mark = {'sig': 'OLD', 'net_ts': 100.0}
        stale = {'rid': 'a', 'ts': 90.0, 'tab': 't', 'type': 'XHR', 'method': 'POST',
                 'url': 'https://shop.example/addresses/v1', 'post': '', 'status': 400, 'mime': 'application/json',
                 'size': 10, 'body': '{"errorMessage":"stale"}', 'note': '', 'done': True}
        fresh = dict(stale, rid='b', ts=101.0, body='{"errorMessage":"Error while adding invoice information."}')
        fine_ = dict(stale, rid='c', ts=101.5, url='https://shop.example/auditrecords/v1', status=200, body='{}')
        for e in (stale, fresh, fine_):
            wa._network_add(e)
        p = tail_patches(['S1', 'S2'], ['OLD'])
        with p[0], p[1], p[2], p[3], p[4], p[5]:
            result = run(wa._async_snapshot_after_action(mark, "browse_act('7')"))
        head = result.split('\n\nS2')[0]
        assert 'POST 400 https://shop.example/addresses/v1' in head
        assert 'Error while adding invoice information' in head
        assert 'stale' not in head, "a failure from before the action is not the action's"
        assert 'auditrecords' not in head, "a request that succeeded is not listed"
        assert result.endswith('\n\nS2')

    def test_a_mark_without_a_moment_flags_without_a_failure_list(self):
        wa = make_wa()
        wa._clear_network_log()
        wa._network_add({'rid': 'a', 'ts': 1.0, 'tab': 't', 'type': 'XHR', 'method': 'POST',
                         'url': 'https://shop.example/x', 'post': '', 'status': 500, 'mime': '',
                         'size': None, 'body': None, 'note': '', 'done': True})
        p = tail_patches(['S1', 'S2'], ['OLD'])
        with p[0], p[1], p[2], p[3], p[4], p[5]:
            result = run(wa._async_snapshot_after_action({'sig': 'OLD'}, "browse_act('7')"))
        assert result.startswith('NOTE: the page looks exactly')
        assert 'shop.example/x' not in result

    def test_verdict_off_keeps_the_waits_but_never_retries_or_flags(self):
        # A click inside an iframe: the parent may still react (a 3-D Secure submit),
        # so the change wait and the settle run — but an unchanged signature proves
        # nothing about the frame, so no second read and no "looks exactly as before".
        wa = make_wa()
        mark = {'sig': 'OLD'}
        snapshots = ['S1', 'S2']
        p = tail_patches(snapshots, ['OLD', 'OLD'])
        with p[0] as change, p[1], p[2], p[3] as settle, p[4], p[5]:
            result = run(wa._async_snapshot_after_action(mark, "the click at (1, 2)", verdict=False))
        assert result == 'S1'
        change.assert_awaited_once_with(mark)
        settle.assert_awaited_once()
        assert snapshots == ['S2'], "the page was read a second time"

    def _after_frame_action(self, frame_before, frame_after):
        wa = make_wa()
        mark = {'sig': 'OLD', 'frame_id': 'FRAME', 'frame_sig': frame_before}
        frame = AsyncMock(side_effect=frame_after) if isinstance(frame_after, Exception) \
            else AsyncMock(return_value={'sig': frame_after})
        p = tail_patches(['S1', 'S2'], ['OLD'])
        with p[0], p[1], p[2], p[3], p[4], p[5], \
                patch.object(WebAccess, '_async_eval_in_frame_target', new=frame):
            return run(wa._async_snapshot_after_action(mark, "browse_act('3')"))

    def test_an_iframe_that_reacted_is_not_flagged_though_the_page_around_it_stood_still(self):
        # The 2026-11 Dintero checkout: "Bekräfta information" inside the payment frame
        # worked, the shop page around it did not move, and the NOTE said nothing happened.
        assert self._after_frame_action('f#10:1', 'f#40:7') == 'S1'

    def test_an_iframe_that_stood_still_with_the_page_is_flagged(self):
        # ...while a Pay button that swallowed the click is exactly what the NOTE is for.
        result = self._after_frame_action('f#10:1', 'f#10:1')
        assert result.startswith('NOTE: the page looks exactly')
        assert result.endswith('\n\nS2')

    @pytest.mark.parametrize("before, after", [
        ('f#10:1', RuntimeError('target closed')),   # the frame navigated away or closed
        (None, 'f#10:1'),                            # it could not be read before the action
    ])
    def test_an_iframe_that_cannot_be_compared_is_not_flagged(self, before, after):
        assert self._after_frame_action(before, after) == 'S1'

    def test_the_mark_of_an_action_inside_an_iframe_carries_the_frames_own_signature(self):
        wa = make_wa()
        frame = AsyncMock(return_value={'sig': 'f#10:1'})
        with probes(probe(sig='MAIN')), patch.object(WebAccess, '_async_eval_in_frame_target', new=frame):
            mark = run(wa._async_action_mark('FRAME'))
            plain = run(wa._async_action_mark())
        assert (mark['sig'], mark['frame_sig']) == ('MAIN', 'f#10:1')
        assert frame.await_args.args[0] == 'FRAME'
        assert 'frame_id' not in plain, "an action on the main page compares the main page alone"

    def test_no_mark_means_no_change_wait_no_settle_no_retry(self):
        wa = make_wa()
        p = tail_patches(['S1'], ['OLD'])
        with p[0] as change, p[1], p[2], p[3] as settle, p[4], p[5]:
            assert run(wa._async_snapshot_after_action(None, "browse_act('3=x')")) == 'S1'
        change.assert_not_awaited()
        settle.assert_not_awaited()

    def test_unknown_mark_signature_skips_the_retry(self):
        wa = make_wa()
        p = tail_patches(['S1'], ['OLD'])
        with p[0], p[1], p[2], p[3], p[4], p[5]:
            assert run(wa._async_snapshot_after_action({'sig': None}, "browse_key('Enter')")) == 'S1'

    def test_settle_gets_the_signature_the_change_wait_saw(self):
        wa = make_wa()
        mark = {'sig': 'OLD'}
        p = tail_patches(['NEW'], ['NEW'], change=('changed', probe(sig='MID')))
        with p[0], p[1], p[2], p[3] as settle, p[4], p[5]:
            run(wa._async_snapshot_after_action(mark, "browse_act('3')"))
        assert settle.await_args.args[1] == 'MID'


@pytest.mark.unit
class TestBrowseActMarks:
    """Only a click or Enter can move the page on: a fills-only batch takes no mark."""

    def _act(self, spec):
        wa = make_wa()
        marks = []

        async def mark(self, frame_id=None):
            marks.append(1)
            return {'sig': 'S'}

        tail = AsyncMock(return_value='SNAP')
        with patch.object(WebAccess, '_async_browse_ready_midsession', new=AsyncMock(return_value=None)), \
                patch.object(WebAccess, '_async_execute_action', new=AsyncMock(return_value=None)), \
                patch.object(WebAccess, '_async_action_mark', new=mark), \
                patch.object(WebAccess, '_async_snapshot_after_action', new=tail):
            result = run(wa._async_browse_act(spec))
        return result, marks, tail

    def test_click_takes_a_mark_and_the_tail_gets_it(self):
        result, marks, tail = self._act('3')
        assert result == 'SNAP'
        assert len(marks) == 1
        assert tail.await_args.args == ({'sig': 'S'}, "browse_act('3')")

    def test_fills_only_batch_takes_no_mark(self):
        _, marks, tail = self._act('3=a; 4=b')
        assert marks == []
        assert tail.await_args.args[0] is None

    def test_fill_with_enter_takes_a_mark(self):
        _, marks, tail = self._act('3=query\\n')
        assert len(marks) == 1
        assert tail.await_args.args[0] is not None

    def test_batch_marks_right_before_the_final_click(self):
        _, marks, _ = self._act('3=a; 4=b; 5')
        assert len(marks) == 1


@pytest.mark.unit
class TestBrowseOpenSettles:

    def test_open_and_snapshot_wait_for_the_page_to_settle(self):
        wa = make_wa()
        with patch.object(WebAccess, '_async_browse_ready_midsession', new=AsyncMock(return_value=None)), \
                patch.object(WebAccess, '_async_sync_popup_targets', new=AsyncMock()), \
                patch.object(WebAccess, '_async_wait_for_page_ready', new=AsyncMock()), \
                patch.object(WebAccess, '_async_wait_for_settle', new=AsyncMock(return_value='settled')) as settle, \
                patch.object(WebAccess, '_async_take_snapshot', new=AsyncMock(return_value='SNAP')):
            assert run(wa._async_browse_snapshot()) == 'SNAP'
        settle.assert_awaited_once()


@pytest.mark.unit
class TestPageContextText:

    def test_alerts_and_text_lines(self):
        ctx = {'h1': 'Sign in', 'alerts': ['Wrong password. Try again'],
               'text': 'Google sent a notification to your phone. Tap Yes to continue.'}
        assert WebAccess._format_page_context(ctx) == (
            'Page: h1 "Sign in"\n'
            'Alerts: "Wrong password. Try again"\n'
            'Text: Google sent a notification to your phone. Tap Yes to continue.\n')

    def test_text_without_headings(self):
        assert WebAccess._format_page_context({'text': 'Loading'}) == 'Text: Loading\n'

    def test_nothing_renders_nothing(self):
        assert WebAccess._format_page_context({}) == ''
        assert WebAccess._format_page_context(None) == ''
