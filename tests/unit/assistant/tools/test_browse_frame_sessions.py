import asyncio
import itertools
import json
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, patch
from nodriver.core.connection import ProtocolException

from assistant.tools.WebAccess import WebAccess


class FakeTargetInfo:
    def __init__(self, target_id, type_='iframe', url=''):
        self.target_id = target_id
        self.type_ = type_
        self.url = url


class FakeTarget:
    """Browser-side target wrapper: nodriver exposes TargetInfo as .target."""
    def __init__(self, target_id, type_='iframe', url=''):
        self.target = FakeTargetInfo(target_id, type_, url)


class FakeBrowser:
    """`live` is what Target.getTargets answers: by default every target of the registry,
    so a test that does not care sees no filtering; a list narrows it, an exception fails it."""
    def __init__(self, targets, update_error=None, live=None):
        self.targets = targets
        self._update_error = update_error
        self._live = live
        self.connection = SimpleNamespace(send=self._send)

    async def update_targets(self):
        if self._update_error:
            raise self._update_error

    async def _send(self, _cdp_call):
        if isinstance(self._live, Exception):
            raise self._live
        return self._live if self._live is not None else [t.target for t in self.targets if t.target]


def make_wa(browser=None, frame_sessions=None):
    wa = WebAccess.__new__(WebAccess)
    wa._browser = browser
    wa._tab = None
    wa._browse_frame_sessions = frame_sessions if frame_sessions is not None else {}
    wa._browse_snapshot_frame_ids = []
    wa._browse_snapshot_labels = []
    wa._pending_target_notes = []
    wa._dialog_notes = []
    return wa


def run(coro):
    return asyncio.run(coro)


@pytest.mark.unit
class TestFrameLabel:

    def test_host_and_path_without_query(self):
        assert WebAccess._frame_label('https://pay.klarna.com/v1/sessions?sid=abc&x=1') \
            == 'pay.klarna.com/v1/sessions'

    def test_trailing_slash_stripped(self):
        assert WebAccess._frame_label('https://example.com/') == 'example.com'

@pytest.mark.unit
class TestAttachFrameSession:

    def test_cached_session_is_reused_without_cdp_roundtrip(self):
        wa = make_wa(frame_sessions={'t1': 'sess-cached'})
        wa._tab = SimpleNamespace(send=AsyncMock())
        assert run(wa._async_attach_frame_session('t1')) == 'sess-cached'
        wa._tab.send.assert_not_awaited()

    def test_attach_on_cache_miss_stores_session(self):
        wa = make_wa()
        wa._tab = SimpleNamespace(send=AsyncMock(return_value='sess-new'))
        assert run(wa._async_attach_frame_session('t1')) == 'sess-new'
        assert wa._browse_frame_sessions == {'t1': 'sess-new'}
        wa._tab.send.assert_awaited_once()


@pytest.mark.unit
class TestEvalInFrameTarget:
    """Retry contract: only ProtocolException (dead session — command never ran)
    triggers one re-attach; JS errors propagate immediately so non-idempotent
    actions like clicks are never re-sent after a successful run."""

    def test_stale_session_reattaches_once(self):
        wa = make_wa(frame_sessions={'t1': 'stale'})
        with patch.object(WebAccess, '_async_attach_frame_session',
                          new=AsyncMock(side_effect=['stale', 'fresh'])) as attach, \
                patch.object(WebAccess, '_evaluate_in_frame',
                             new=AsyncMock(side_effect=[ProtocolException('dead session'), 'ok'])):
            assert run(wa._async_eval_in_frame_target('t1', 'x')) == 'ok'
        assert attach.await_count == 2
        assert 't1' not in wa._browse_frame_sessions  # stale entry evicted

    def test_dead_target_raises_after_second_failure(self):
        wa = make_wa(frame_sessions={'t1': 'stale'})
        with patch.object(WebAccess, '_async_attach_frame_session',
                          new=AsyncMock(return_value='s')), \
                patch.object(WebAccess, '_evaluate_in_frame',
                             new=AsyncMock(side_effect=ProtocolException('gone'))):
            with pytest.raises(ProtocolException):
                run(wa._async_eval_in_frame_target('t1', 'x'))
        assert 't1' not in wa._browse_frame_sessions

    def test_js_exception_propagates_without_retry(self):
        wa = make_wa(frame_sessions={'t1': 's1'})
        with patch.object(WebAccess, '_async_attach_frame_session',
                          new=AsyncMock(return_value='s1')) as attach, \
                patch.object(WebAccess, '_evaluate_in_frame',
                             new=AsyncMock(side_effect=RuntimeError('frame JS threw: boom'))):
            with pytest.raises(RuntimeError):
                run(wa._async_eval_in_frame_target('t1', 'el.click()'))
        assert attach.await_count == 1
        assert wa._browse_frame_sessions == {'t1': 's1'}  # session still trusted


@pytest.mark.unit
class TestEvaluateInFrame:

    def _eval(self, send_result):
        wa = make_wa()
        with patch.object(WebAccess, '_send_on_session', new=AsyncMock(return_value=send_result)):
            return run(wa._evaluate_in_frame('s1', 'expr'))

    def test_returns_remote_object_value(self):
        assert self._eval((SimpleNamespace(value=7), None)) == 7

    def test_returns_none_without_remote_object(self):
        assert self._eval((None, None)) is None

    def test_js_exception_description_raised(self):
        errors = SimpleNamespace(
            exception=SimpleNamespace(description='TypeError: x is not a function'), text=None)
        with pytest.raises(RuntimeError, match='TypeError: x is not a function'):
            self._eval((None, errors))

    def test_falls_back_to_error_text(self):
        errors = SimpleNamespace(exception=None, text='Uncaught')
        with pytest.raises(RuntimeError, match='Uncaught'):
            self._eval((None, errors))


@pytest.mark.unit
class TestListFrameTargets:

    def test_filters_pages_blanks_and_skip_patterns(self):
        targets = [
            FakeTarget('page1', type_='page', url='https://shop.example'),
            FakeTarget('blank', url='about:blank'),
            FakeTarget('empty', url=''),
            FakeTarget('maps', url='https://google.com/maps/embed?x=1'),
            FakeTarget('klarna', url='https://pay.klarna.com/frame'),
            SimpleNamespace(target=None),
        ]
        wa = make_wa(browser=FakeBrowser(targets))
        result = run(wa._async_list_frame_targets())
        assert [ti.target_id for ti in result] == ['klarna']

    def test_skips_stale_iframe_target_that_is_the_page_itself(self):
        # browse_open() on a URL that was an iframe of the previous page: Chrome
        # still lists the dead OOPIF target under that URL and every command to it
        # burns _FRAME_SEND_TIMEOUT — it must be skipped, other frames kept.
        targets = [
            FakeTarget('stale', url='https://book.example.com/book?id=abc12&lang=se'),
            FakeTarget('live', url='https://pay.klarna.com/frame'),
        ]
        wa = make_wa(browser=FakeBrowser(targets))
        wa._tab = SimpleNamespace(target=SimpleNamespace(
            url='https://book.example.com/book/?id=abc12&lang=se&mode=embed'))
        result = run(wa._async_list_frame_targets())
        assert [ti.target_id for ti in result] == ['live']

    def test_iframe_targets_kept_when_no_tab(self):
        wa = make_wa(browser=FakeBrowser([FakeTarget('f', url='https://pay.klarna.com/frame')]))
        assert [ti.target_id for ti in run(wa._async_list_frame_targets())] == ['f']

    def test_prunes_sessions_of_vanished_targets(self):
        wa = make_wa(
            browser=FakeBrowser([FakeTarget('live', url='https://pay.klarna.com/frame')]),
            frame_sessions={'live': 's1', 'gone': 's2'},
        )
        run(wa._async_list_frame_targets())
        assert wa._browse_frame_sessions == {'live': 's1'}

    def test_update_targets_failure_returns_empty(self):
        wa = make_wa(browser=FakeBrowser([], update_error=RuntimeError('closed')))
        assert run(wa._async_list_frame_targets()) == []

    def test_closed_target_nodriver_still_lists_is_skipped_and_its_session_dropped(self):
        # nodriver's registry never drops a closed target: a tracking pixel's frame
        # from an earlier page would be attached to, and timed out on, at every
        # snapshot of the session. Chrome's own list is authoritative.
        pixel = FakeTarget('pixel', url='https://pixel.example/tr?id=1')
        card = FakeTarget('card', url='https://js.stripe.com/v3/elements-inner-card.html')
        wa = make_wa(browser=FakeBrowser([pixel, card], live=[card.target]),
                     frame_sessions={'pixel': 's1', 'card': 's2'})
        result = run(wa._async_list_frame_targets())
        assert [ti.target_id for ti in result] == ['card']
        assert wa._browse_frame_sessions == {'card': 's2'}

    def test_get_targets_failure_keeps_nodriver_list(self):
        targets = [FakeTarget('a', url='https://pay.klarna.com/frame'),
                   FakeTarget('b', url='https://js.stripe.com/v3/elements-inner-card.html')]
        wa = make_wa(browser=FakeBrowser(targets, live=RuntimeError('transport gone')))
        assert [ti.target_id for ti in run(wa._async_list_frame_targets())] == ['a', 'b']


@pytest.mark.unit
class TestSnapshotFrameMerging:

    def _snapshot(self, wa, frame_targets, frame_eval):
        async def fake_eval(_tab, expr):
            if 'document.title' in expr:
                return 'Checkout'
            return {'records': [{'tag': 'button', 'label': 'Main'}],
                    'crossOriginFrames': ['https://pay.klarna.com/frame']}

        with patch.object(WebAccess, '_evaluate', side_effect=fake_eval), \
                patch.object(WebAccess, '_async_sync_popup_targets', new=AsyncMock()), \
                patch.object(WebAccess, '_async_list_frame_targets',
                             new=AsyncMock(return_value=frame_targets)), \
                patch.object(WebAccess, '_async_eval_in_frame_target', new=frame_eval):
            return run(wa._async_take_snapshot())

    def test_frame_elements_merge_with_offset_indices(self):
        wa = make_wa()
        wa._tab = SimpleNamespace(target=FakeTargetInfo('main', 'page', 'https://shop.example/kassa'))
        frame = FakeTargetInfo('ft1', url='https://pay.klarna.com/frame')
        frame_eval = AsyncMock(return_value={'records': [
            {'tag': 'button', 'label': 'Pay now', 'source': 'pay.klarna.com/frame'}]})

        result = self._snapshot(wa, [frame], frame_eval)

        assert '[0] button "Main"' in result
        assert '[1] button "Pay now"  [in pay.klarna.com/frame]' in result
        assert wa._browse_snapshot_frame_ids == [None, 'ft1']
        assert wa._browse_snapshot_labels == ['Main', 'Pay now']
        # collector ran inside the frame with the main page's record count as offset
        assert ')(1, ' in frame_eval.await_args.args[1]
        # frame was reached, so it must not be reported as hidden
        assert 'Cross-origin iframes on this page' not in result

    def test_unreachable_frame_reported_hidden(self):
        wa = make_wa()
        wa._tab = SimpleNamespace(target=FakeTargetInfo('main', 'page', 'https://shop.example/kassa'))
        frame = FakeTargetInfo('ft1', url='https://pay.klarna.com/frame')
        frame_eval = AsyncMock(side_effect=ProtocolException('no such target'))

        result = self._snapshot(wa, [frame], frame_eval)

        assert wa._browse_snapshot_frame_ids == [None]
        assert 'Cross-origin iframes on this page' in result
        assert 'https://pay.klarna.com/frame' in result

    def test_the_snapshot_leaves_browse_network_the_hosts_it_cannot_capture(self):
        """Each cross-origin host once, however its frames are addressed; gone with the frames."""
        wa = make_wa()
        wa._tab = SimpleNamespace(target=FakeTargetInfo('main', 'page', 'https://shop.example/kassa'))
        pages = [{'records': [], 'crossOriginFrames': ['https://pay.example/card', '//pay.example/3ds', '',
                                                         'https://card.bank.example/form?session=1']},
                 {'records': [{'tag': 'button', 'label': 'Main'}], 'crossOriginFrames': []}]

        async def fake_eval(_tab, expr):
            return 'Checkout' if 'document.title' in expr else pages[0]

        with patch.object(WebAccess, '_evaluate', side_effect=fake_eval), \
                patch.object(WebAccess, '_async_sync_popup_targets', new=AsyncMock()), \
                patch.object(WebAccess, '_async_list_frame_targets', new=AsyncMock(return_value=[])), \
                patch.object(WebAccess, '_OOPIF_SETTLE_TIMEOUT', 0):
            run(wa._async_take_snapshot())
            assert wa._browse_cross_origin_frame_hosts == ['card.bank.example', 'pay.example']
            pages.pop(0)
            run(wa._async_take_snapshot())
        assert wa._browse_cross_origin_frame_hosts == []


@pytest.mark.unit
class TestExecuteActionFrameRouting:

    META = {'tag': 'button', 'type': '', 'role': '', 'contenteditable': None, 'checked': None}

    def test_frame_element_evaluates_in_owning_frame(self):
        wa = make_wa()
        wa._tab = SimpleNamespace()
        wa._browse_snapshot_frame_ids = [None, 'ft9']
        # meta, scroll, click point (no box), JS click
        frame_eval = AsyncMock(side_effect=[self.META, None, {'reason': 'no box'}, True])
        with patch.object(WebAccess, '_async_eval_in_frame_target', new=frame_eval), \
                patch.object(WebAccess, '_evaluate', new=AsyncMock()) as main_eval, \
                patch.object(WebAccess, '_async_dispatch_mouse', new=AsyncMock()) as mouse:
            assert run(wa._async_execute_action(1, None)) is None
        assert all(call.args[0] == 'ft9' for call in frame_eval.await_args_list)
        main_eval.assert_not_awaited()
        mouse.assert_not_awaited()

    def test_a_frame_element_the_real_click_reached_gets_no_js_click_on_top(self):
        """A Pay button pressed twice is an order paid twice."""
        wa = make_wa()
        wa._tab = SimpleNamespace()
        wa._browse_snapshot_frame_ids = [None, 'ft9']
        frame_eval = AsyncMock(side_effect=[self.META, None])   # meta, scroll — and nothing after
        with patch.object(WebAccess, '_async_eval_in_frame_target', new=frame_eval), \
                patch.object(WebAccess, '_async_trusted_frame_click', new=AsyncMock(return_value=True)) as real, \
                patch.object(WebAccess, '_async_trusted_click', new=AsyncMock()) as main_click:
            assert run(wa._async_execute_action(1, None)) is None
        assert real.await_args.args[0] == 'ft9'
        assert frame_eval.await_count == 2
        main_click.assert_not_awaited()

    @pytest.mark.parametrize("field", [
        {'type': 'password', 'hint': 'pw  current-password  '},
        {'type': 'text', 'hint': 'cardSecurityCode   CVC '},
        {'type': 'tel', 'hint': '  cc-exp MM / ÅÅ '},
    ])
    def test_a_fill_of_a_card_or_credential_field_never_logs_what_was_typed(self, field):
        wa = make_wa()
        wa._tab = SimpleNamespace()
        wa._browse_snapshot_frame_ids = [None]
        meta = {**self.META, 'tag': 'input', **field}

        async def ev(_tab, expr):
            if meta is not None and 'getAttribute' in expr:
                return meta
            raise RuntimeError('the page is gone')

        logged = []
        with patch.object(WebAccess, '_evaluate', side_effect=ev), \
                patch('assistant.tools.WebAccess.info', side_effect=lambda *a, **k: logged.append(str(a))), \
                patch('assistant.tools.WebAccess.trace', side_effect=lambda *a, **k: logged.append(str(a))), \
                patch('assistant.tools.WebAccess.note', side_effect=lambda *a, **k: logged.append(str(a))):
            outcome = run(wa._async_execute_action(0, '7731'))
        assert any('fill [0]' in line for line in logged)
        assert not any('7731' in line for line in logged)
        assert '7731' not in (outcome or '')

    def test_main_page_element_evaluates_on_tab(self):
        wa = make_wa()
        wa._tab = SimpleNamespace()
        wa._browse_snapshot_frame_ids = [None]
        # meta, scroll, click point (no box), JS click
        with patch.object(WebAccess, '_evaluate',
                          new=AsyncMock(side_effect=[self.META, None, {'reason': 'no box'}, True])) as main_eval, \
                patch.object(WebAccess, '_async_eval_in_frame_target', new=AsyncMock()) as frame_eval:
            assert run(wa._async_execute_action(0, None)) is None
        assert main_eval.await_count == 4
        frame_eval.assert_not_awaited()

    def test_visible_main_page_element_gets_a_real_mouse_click(self):
        wa = make_wa()
        wa._tab = SimpleNamespace()
        wa._browse_snapshot_frame_ids = [None]
        main_eval = AsyncMock(side_effect=[self.META, None, {'x': 120.0, 'y': 48.5}])
        with patch.object(WebAccess, '_evaluate', new=main_eval), \
                patch.object(WebAccess, '_async_dispatch_click_at', new=AsyncMock()) as mouse:
            assert run(wa._async_execute_action(0, None)) is None
        mouse.assert_awaited_once_with(120.0, 48.5)
        # the trusted click is the click: no e.click() evaluation follows it
        assert not any('click()' in call.args[1] for call in main_eval.await_args_list)

    def test_failed_mouse_click_falls_back_to_js_click(self):
        wa = make_wa()
        wa._tab = SimpleNamespace()
        wa._browse_snapshot_frame_ids = [None]
        main_eval = AsyncMock(side_effect=[self.META, None, {'x': 10.0, 'y': 10.0}, True])
        with patch.object(WebAccess, '_evaluate', new=main_eval), \
                patch.object(WebAccess, '_async_dispatch_click_at',
                             new=AsyncMock(side_effect=TimeoutError())):
            assert run(wa._async_execute_action(0, None)) is None
        assert 'click()' in main_eval.await_args_list[-1].args[1]

    def test_failed_click_point_lookup_falls_back_to_js_click(self):
        wa = make_wa()
        wa._tab = SimpleNamespace()
        wa._browse_snapshot_frame_ids = [None]
        # meta, scroll, click point (the page's context went away mid-lookup), JS click
        main_eval = AsyncMock(side_effect=[self.META, None, ProtocolException('context destroyed'), True])
        with patch.object(WebAccess, '_evaluate', new=main_eval), \
                patch.object(WebAccess, '_async_dispatch_click_at', new=AsyncMock()) as mouse:
            assert run(wa._async_execute_action(0, None)) is None
        mouse.assert_not_awaited()
        assert 'click()' in main_eval.await_args_list[-1].args[1]

    COVERED = {'reason': 'covered', 'hit': 'aside#usercentrics-cmp-ui', 'shadow': True}

    @staticmethod
    def covered_wa():
        wa = make_wa()
        wa._tab = SimpleNamespace()
        wa._browse_snapshot_frame_ids = [None]
        wa._UNCOVER_STEP = 0.0  # no real waiting for the banner to close
        return wa

    def test_covered_element_gets_a_consent_pass_and_then_a_real_click(self):
        # A consent wall mounted after browse_open's pass (fedex.com, #82): answered on
        # the spot, and the click lands on the element it uncovered.
        wa = self.covered_wa()
        # meta, scroll, click point (covered), location.href, click point again
        main_eval = AsyncMock(side_effect=[self.META, None, self.COVERED,
                                           'https://parcels.example/track', {'x': 40.0, 'y': 20.0}])
        consent = AsyncMock(return_value={'clicked': 1, 'removed': 0})
        with patch.object(WebAccess, '_evaluate', new=main_eval), \
                patch.object(WebAccess, '_async_remove_cookies_banner', new=consent), \
                patch.object(WebAccess, '_async_dispatch_click_at', new=AsyncMock()) as mouse:
            assert run(wa._async_execute_action(0, None)) is None
        consent.assert_awaited_once()
        mouse.assert_awaited_once_with(40.0, 20.0)

    def test_covered_element_with_no_banner_to_answer_falls_back_to_js_click(self):
        wa = self.covered_wa()
        main_eval = AsyncMock(side_effect=[self.META, None, self.COVERED,
                                           'https://parcels.example/track', True])
        consent = AsyncMock(return_value={'clicked': 0, 'removed': 0})
        with patch.object(WebAccess, '_evaluate', new=main_eval), \
                patch.object(WebAccess, '_async_remove_cookies_banner', new=consent), \
                patch.object(WebAccess, '_async_dispatch_click_at', new=AsyncMock()) as mouse:
            assert run(wa._async_execute_action(0, None)) is None
        consent.assert_awaited_once()
        mouse.assert_not_awaited()
        assert 'click()' in main_eval.await_args_list[-1].args[1]

    def test_covered_element_on_an_auth_page_gets_no_consent_pass(self):
        # a stray consent click on a login flow submits it, so the wall stays
        wa = self.covered_wa()
        main_eval = AsyncMock(side_effect=[self.META, None, self.COVERED,
                                           'https://id.example/signin?next=/track', True])
        consent = AsyncMock(return_value={'clicked': 1, 'removed': 0})
        with patch.object(WebAccess, '_evaluate', new=main_eval), \
                patch.object(WebAccess, '_async_remove_cookies_banner', new=consent), \
                patch.object(WebAccess, '_async_dispatch_click_at', new=AsyncMock()) as mouse:
            assert run(wa._async_execute_action(0, None)) is None
        consent.assert_not_awaited()
        mouse.assert_not_awaited()

    def test_other_click_point_verdicts_skip_the_consent_pass(self):
        wa = self.covered_wa()
        main_eval = AsyncMock(side_effect=[self.META, None, {'reason': 'inside an iframe'}, True])
        consent = AsyncMock(return_value={'clicked': 1, 'removed': 0})
        with patch.object(WebAccess, '_evaluate', new=main_eval), \
                patch.object(WebAccess, '_async_remove_cookies_banner', new=consent), \
                patch.object(WebAccess, '_async_dispatch_click_at', new=AsyncMock()) as mouse:
            assert run(wa._async_execute_action(0, None)) is None
        consent.assert_not_awaited()
        mouse.assert_not_awaited()
        assert 'click()' in main_eval.await_args_list[-1].args[1]

    def test_element_gone_after_the_consent_pass_reports_the_reload(self):
        # fedex.com reloads the page once the wall is answered (#82): the index is stale
        # because of the consent, and the LLM is told so rather than "element gone"
        wa = self.covered_wa()
        main_eval = AsyncMock(side_effect=[self.META, None, self.COVERED,
                                           'https://parcels.example/track',
                                           {'reason': 'element gone'}, False])
        consent = AsyncMock(return_value={'clicked': 1, 'removed': 0})
        with patch.object(WebAccess, '_evaluate', new=main_eval), \
                patch.object(WebAccess, '_async_remove_cookies_banner', new=consent), \
                patch.object(WebAccess, '_async_dispatch_click_at', new=AsyncMock()):
            assert run(wa._async_execute_action(0, None)) == WebAccess.CONSENT_RELOAD_ERROR

    def test_banner_that_stays_up_is_waited_for_then_given_up_on(self):
        wa = self.covered_wa()
        wa._UNCOVER_WAIT = 0.0
        main_eval = AsyncMock(side_effect=[self.META, None, self.COVERED,
                                           'https://parcels.example/track', self.COVERED, True])
        consent = AsyncMock(return_value={'clicked': 1, 'removed': 0})
        with patch.object(WebAccess, '_evaluate', new=main_eval), \
                patch.object(WebAccess, '_async_remove_cookies_banner', new=consent), \
                patch.object(WebAccess, '_async_dispatch_click_at', new=AsyncMock()) as mouse:
            assert run(wa._async_execute_action(0, None)) is None
        mouse.assert_not_awaited()
        assert 'click()' in main_eval.await_args_list[-1].args[1]

    def test_click_reports_element_gone_when_finder_misses(self):
        wa = make_wa()
        wa._tab = SimpleNamespace()
        wa._browse_snapshot_frame_ids = [None]
        # meta resolves, but by click time the element has left the DOM
        with patch.object(WebAccess, '_evaluate',
                          new=AsyncMock(side_effect=[self.META, None, None, False])):
            assert run(wa._async_execute_action(0, None)) == 'element no longer in DOM'

    def test_missing_meta_reports_element_gone(self):
        wa = make_wa()
        wa._tab = SimpleNamespace()
        wa._browse_snapshot_frame_ids = [None]
        with patch.object(WebAccess, '_evaluate', new=AsyncMock(return_value=None)):
            assert run(wa._async_execute_action(0, None)) == 'element no longer in DOM'



@pytest.mark.unit
class TestTrustedFrameClick:
    """A button inside a cross-origin iframe gets a real mouse click (the 2026-11 card form
    whose Pay button ignored the JS click): the element's centre in its frame plus where the
    frame stands on the page, and nothing pressed until the frame itself reports the pointer
    on the element."""

    @pytest.fixture(autouse=True)
    def fast(self, monkeypatch):
        monkeypatch.setattr(WebAccess, '_FRAME_POSITION_STEP', 0)
        monkeypatch.setattr(WebAccess, '_FRAME_HOVER_STEP', 0)
        monkeypatch.setattr(WebAccess, '_FRAME_HOVER_WAIT', 0.01)

    def click(self, origins, hover, points=None):
        """Run the click with the frame standing at each of `origins` in turn (the last one
        repeating), the element at each of `points` in its frame, and the frame answering the
        hover check with `hover`. Returns (clicked, the mouse events sent)."""
        wa = make_wa()
        origins, points = list(origins), list(points or [{'x': 40.0, 'y': 12.0}])
        events = []

        async def ev(js):
            if js == WebAccess._FRAME_HOVER_READ_JS:
                return hover
            if WebAccess._FRAME_HOVER_ARM_JS in js:
                return True
            return points.pop(0) if len(points) > 1 else points[0]

        async def origin(self, frame_id):
            return origins.pop(0) if len(origins) > 1 else origins[0]

        async def mouse(self, type_, x, y, button=None):
            events.append((type_, x, y))

        with patch.object(WebAccess, '_async_frame_origin', new=origin), \
                patch.object(WebAccess, '_async_dispatch_mouse', new=mouse):
            return run(wa._async_trusted_frame_click('ft9', ev, 'FIND')), events

    def test_the_click_lands_at_the_frames_position_plus_the_elements_point_in_it(self):
        clicked, events = self.click([(226.0, 274.5)], {'trusted': True, 'on': True})
        assert clicked
        assert [e[0] for e in events][-2:] == ['mousePressed', 'mouseReleased']
        assert {e[1:] for e in events} == {(266.0, 286.5)}

    def test_a_pointer_the_frame_sees_on_another_element_presses_nothing(self):
        clicked, events = self.click([(226.0, 274.5)], {'trusted': True, 'on': False})
        assert not clicked
        assert {e[0] for e in events} == {'mouseMoved'}

    def test_a_pointer_the_frame_never_sees_presses_nothing(self):
        # an overlay on the page above the frame, or a position that is simply wrong
        clicked, events = self.click([(226.0, 274.5)], None)
        assert not clicked
        assert {e[0] for e in events} == {'mouseMoved'}

    def test_a_frame_still_scrolling_into_view_is_clicked_where_it_comes_to_rest(self):
        clicked, events = self.click([(226.0, 900.0), (226.0, 500.0), (226.0, 397.5)],
                                     {'trusted': True, 'on': True})
        assert clicked
        assert {e[1:] for e in events} == {(266.0, 409.5)}

    def test_a_frame_that_never_comes_to_rest_is_left_to_the_js_click(self):
        moving = [(226.0, float(y)) for y in range(900, 900 - 10 * WebAccess._FRAME_POSITION_READS, -10)]
        clicked, events = self.click(moving, {'trusted': True, 'on': True})
        assert not clicked and events == []

    @pytest.mark.parametrize("point, origin", [
        ({'reason': 'covered', 'hit': 'div.modal'}, (226.0, 274.5)),   # hidden inside its own frame
        ({'x': 40.0, 'y': 12.0}, None),                                 # the frame cannot be placed
    ])
    def test_an_element_or_frame_that_cannot_be_placed_sends_no_mouse_event(self, point, origin):
        clicked, events = self.click([origin], {'trusted': True, 'on': True}, points=[point])
        assert not clicked and events == []


@pytest.mark.unit
class TestFrameOrigin:
    """Where an iframe target's viewport begins on the page: the content box of the <iframe>
    element owning it, asked of whichever document holds that element, summed through the
    frames it is nested in."""

    def origin(self, boxes, targets=('outer', 'inner', 'sibling')):
        """`boxes` maps (document asked, frame owned) to the owner's top-left; the page is None."""
        wa = make_wa()
        asked = []

        async def owner_box(self, parent, frame_id):
            asked.append((parent, frame_id))
            return boxes.get((parent, frame_id))

        listed = AsyncMock(return_value=[SimpleNamespace(target_id=t) for t in targets])
        with patch.object(WebAccess, '_async_frame_owner_box', new=owner_box), \
                patch.object(WebAccess, '_async_list_frame_targets', new=listed):
            return run(wa._async_frame_origin('inner')), asked, listed

    def test_a_frame_of_the_page_itself_needs_no_other_frame(self):
        origin, _, listed = self.origin({(None, 'inner'): (163.0, 161.5)})
        assert origin == (163.0, 161.5)
        listed.assert_not_awaited()

    def test_a_nested_frame_adds_the_position_of_every_frame_around_it(self):
        origin, asked, _ = self.origin({('outer', 'inner'): (63.0, 113.0), (None, 'outer'): (163.0, 161.5)})
        assert origin == (226.0, 274.5)
        assert ('inner', 'inner') not in asked, "a frame is never asked for its own owner"

    def test_a_frame_no_document_owns_has_no_position(self):
        origin, _, _ = self.origin({})
        assert origin is None

    def test_a_chain_that_never_reaches_the_page_gives_up(self):
        # two frames each claiming to hold the other: no position rather than a loop
        origin, _, _ = self.origin({('outer', 'inner'): (1.0, 1.0), ('inner', 'outer'): (1.0, 1.0)})
        assert origin is None


@pytest.mark.unit
class TestExtractFrameTargetTexts:

    def test_collects_labels_texts_and_skips_failures(self):
        wa = make_wa()
        frames = [
            FakeTargetInfo('dead', url='https://a.example/widget'),
            FakeTargetInfo('live', url='https://pay.klarna.com/frame?sid=1'),
        ]
        long_text = 'Betala med Klarna. ' * 5

        async def fake_frame_eval(_self, target_id, expr):
            if target_id == 'dead':
                raise ProtocolException('no such target')
            return '<body><p>html</p></body>'

        with patch.object(WebAccess, '_async_list_frame_targets',
                          new=AsyncMock(return_value=frames)), \
                patch.object(WebAccess, '_async_eval_in_frame_target', new=fake_frame_eval), \
                patch.object(WebAccess, 'extract_text_with_links', return_value=long_text):
            result = run(wa._async_extract_frame_target_texts())

        assert result.startswith('[iframe pay.klarna.com/frame]')
        assert long_text.strip() in result
        assert 'a.example' not in result

    def test_short_or_empty_frames_yield_nothing(self):
        wa = make_wa()
        frames = [FakeTargetInfo('live', url='https://pay.klarna.com/frame')]
        with patch.object(WebAccess, '_async_list_frame_targets',
                          new=AsyncMock(return_value=frames)), \
                patch.object(WebAccess, '_async_eval_in_frame_target',
                             new=AsyncMock(return_value='<body>x</body>')), \
                patch.object(WebAccess, 'extract_text_with_links', return_value='tiny'):
            assert run(wa._async_extract_frame_target_texts()) == ''


class FakeSessionConnection:
    """Captures the hand-built session-scoped message and resolves the Transaction."""

    def __init__(self, closed=False):
        self.closed = closed
        self.connected = False
        self.mapper = {}
        self.__count__ = itertools.count(11)
        self.sent_payloads = []
        outer = self

        class Ws:
            async def send(self, payload):
                msg = json.loads(payload)
                outer.sent_payloads.append(msg)
                outer.mapper[msg['id']].set_result('cdp-result')

        self.websocket = Ws()

    async def connect(self):
        self.connected = True
        self.closed = False


def fake_cdp_command():
    yield {'method': 'Fake.command', 'params': {'a': 1}}


@pytest.mark.unit
class TestSendOnSession:

    def test_message_carries_session_id_and_fresh_id(self):
        wa = make_wa()
        conn = FakeSessionConnection()
        wa._tab = conn
        result = run(wa._send_on_session(fake_cdp_command(), 'sess-42'))
        assert result == 'cdp-result'
        [msg] = conn.sent_payloads
        assert msg == {'method': 'Fake.command', 'params': {'a': 1},
                       'id': 11, 'sessionId': 'sess-42'}
        assert not conn.connected  # no reconnect when already open

    def test_reconnects_closed_connection_first(self):
        wa = make_wa()
        conn = FakeSessionConnection(closed=True)
        wa._tab = conn
        run(wa._send_on_session(fake_cdp_command(), 'sess-42'))
        assert conn.connected


@pytest.mark.unit
class TestBlankFrameTargets:

    def test_uncommitted_iframe_targets_are_recorded_not_forgotten(self):
        # A fresh OOPIF shows up with an empty or about:blank URL first; it is
        # skipped for collection but the snapshot's settle wait must see it.
        targets = [
            FakeTarget('blank', url='about:blank'),
            FakeTarget('empty', url=''),
            FakeTarget('live', url='https://js.stripe.com/v3/controller-abc.html'),
        ]
        wa = make_wa(browser=FakeBrowser(targets))
        result = run(wa._async_list_frame_targets())
        assert [ti.target_id for ti in result] == ['live']
        assert wa._blank_frame_target_ids == {'blank', 'empty'}


@pytest.mark.unit
class TestSettledFrameTargets:
    CARD = 'https://js.stripe.com/v3/elements-inner-card-1.html#locale=sv'
    CTRL = 'https://js.stripe.com/v3/controller-abc.html'

    def _settle(self, wa, listings):
        calls = iter(listings)

        async def fake_list(_self):
            frames, blank = next(calls)
            _self._blank_frame_target_ids = frozenset(blank)
            return frames

        with patch.object(WebAccess, '_async_list_frame_targets', new=fake_list), \
                patch.object(WebAccess, '_OOPIF_SETTLE_POLL', 0), \
                patch.object(WebAccess, '_OOPIF_SETTLE_TIMEOUT', 5.0):
            return run(wa._async_settled_frame_targets([self.CARD, self.CTRL]))

    def test_no_wait_when_every_src_has_a_target(self):
        wa = make_wa()
        frames = [FakeTargetInfo('c', url=self.CTRL), FakeTargetInfo('k', url=self.CARD)]
        assert self._settle(wa, [(frames, [])]) == frames

    def test_waits_for_the_card_frame_target_to_appear(self):
        # The controller frame exists at once (same host!); the card frame's
        # target arrives two polls later — the snapshot must include it.
        wa = make_wa()
        ctrl = FakeTargetInfo('c', url=self.CTRL)
        card = FakeTargetInfo('k', url=self.CARD)
        result = self._settle(wa, [([ctrl], []), ([ctrl], ['k']), ([ctrl, card], [])])
        assert [ti.target_id for ti in result] == ['c', 'k']

    def test_waits_while_a_target_has_no_url_yet(self):
        wa = make_wa()
        ctrl = FakeTargetInfo('c', url=self.CTRL)
        card = FakeTargetInfo('k', url=self.CARD)
        result = self._settle(wa, [([ctrl, card], ['x']), ([ctrl, card], [])])
        assert [ti.target_id for ti in result] == ['c', 'k']

    def test_gives_up_at_deadline_and_never_waits_for_that_src_again(self):
        wa = make_wa()
        ctrl = FakeTargetInfo('c', url=self.CTRL)
        listed = []

        async def fake_list(_self):
            listed.append(1)
            _self._blank_frame_target_ids = frozenset()
            return [ctrl]

        with patch.object(WebAccess, '_async_list_frame_targets', new=fake_list), \
                patch.object(WebAccess, '_OOPIF_SETTLE_POLL', 0), \
                patch.object(WebAccess, '_OOPIF_SETTLE_TIMEOUT', 0.0):
            assert run(wa._async_settled_frame_targets([self.CARD, self.CTRL])) == [ctrl]
            first = len(listed)
            assert 'js.stripe.com/v3/elements-inner-card-1.html' in wa._oopif_settle_giveups
            # second snapshot: the unresolvable src is memoized — a single listing, no polling
            assert run(wa._async_settled_frame_targets([self.CARD, self.CTRL])) == [ctrl]
        assert len(listed) == first + 1


@pytest.mark.unit
class TestHiddenFramesPerFrame:

    def test_unreached_frame_on_an_attached_host_is_still_reported(self):
        card = 'https://js.stripe.com/v3/elements-inner-card-1.html'
        ctrl = 'https://js.stripe.com/v3/controller-abc.html'

        async def fake_eval(_tab, expr):
            if 'document.title' in expr:
                return 'Betalning'
            return {'records': [{'tag': 'button', 'label': 'Slutför'}],
                    'crossOriginFrames': [ctrl, card]}

        wa = make_wa()
        wa._tab = SimpleNamespace(target=FakeTargetInfo('main', 'page', 'https://bord.example/pay'))
        with patch.object(WebAccess, '_evaluate', side_effect=fake_eval), \
                patch.object(WebAccess, '_async_sync_popup_targets', new=AsyncMock()), \
                patch.object(WebAccess, '_async_settled_frame_targets',
                             new=AsyncMock(return_value=[FakeTargetInfo('c', url=ctrl)])), \
                patch.object(WebAccess, '_async_eval_in_frame_target',
                             new=AsyncMock(return_value={'records': []})):
            result = run(wa._async_take_snapshot())

        assert 'Cross-origin iframes on this page' in result
        assert card in result
        assert ctrl not in result.split('Cross-origin iframes on this page')[1]
        assert 'browse_snapshot()' in result


@pytest.mark.unit
class TestOopifFill:
    META = {'tag': 'input', 'type': 'text', 'role': '', 'contenteditable': None, 'checked': None}

    def _fill(self, wa, frame_eval, focus_in_iframe, value='4242424242424242'):
        wa._tab = SimpleNamespace(send=AsyncMock(return_value=None))
        wa._browse_snapshot_frame_ids = [None, 'card']
        with patch.object(WebAccess, '_async_eval_in_frame_target', new=frame_eval), \
                patch.object(WebAccess, '_evaluate', new=AsyncMock(return_value=focus_in_iframe)):
            return run(wa._async_execute_action(1, value))

    @staticmethod
    def _frame_js_calls(frame_eval):
        return [c.args[1] for c in frame_eval.await_args_list]

    def test_types_via_main_tab_insert_text_when_the_iframe_holds_focus(self):
        wa = make_wa()
        # meta, scroll, focus+clear, read back, commit
        frame_eval = AsyncMock(side_effect=[self.META, None, True, '4242 4242 4242 4242', True])
        assert self._fill(wa, frame_eval, focus_in_iframe=True) is None
        wa._tab.send.assert_awaited_once()
        assert not any('dispatchEvent(new Event(\'change\'' in js and 'setter.call(el, value)' in js
                       for js in self._frame_js_calls(frame_eval))

    def test_falls_back_to_js_fill_when_focus_is_not_in_an_iframe(self):
        wa = make_wa()
        frame_eval = AsyncMock(side_effect=[self.META, None, True, True])  # meta, scroll, focus, legacy
        assert self._fill(wa, frame_eval, focus_in_iframe=False) is None
        wa._tab.send.assert_not_awaited()
        assert any('setter.call(el, value)' in js for js in self._frame_js_calls(frame_eval))

    def test_falls_back_to_js_fill_when_the_field_stays_empty(self):
        wa = make_wa()
        # meta, scroll, focus+clear, read back (empty), legacy fill
        frame_eval = AsyncMock(side_effect=[self.META, None, True, '', True])
        assert self._fill(wa, frame_eval, focus_in_iframe=True) is None
        wa._tab.send.assert_awaited_once()
        assert any('setter.call(el, value)' in js for js in self._frame_js_calls(frame_eval))


@pytest.mark.unit
class TestBrowseSnapshotCommand:

    def _wa(self, session='https://shop.example/kassa'):
        wa = make_wa()
        wa._browse_session_url = session
        wa._browse_pdf_text = None
        wa._tab = SimpleNamespace()
        return wa

    def test_waits_for_the_page_then_snapshots_without_acting(self):
        wa = self._wa()
        with patch.object(WebAccess, '_async_browse_ready_midsession', new=AsyncMock(return_value=None)), \
                patch.object(WebAccess, '_async_sync_popup_targets', new=AsyncMock()) as sync, \
                patch.object(WebAccess, '_async_wait_for_page_ready', new=AsyncMock()) as ready, \
                patch.object(WebAccess, '_async_take_snapshot', new=AsyncMock(return_value='SNAP')):
            assert run(wa._async_browse_snapshot()) == 'SNAP'
        sync.assert_awaited_once()
        ready.assert_awaited_once()


@pytest.mark.unit
class TestBrowseClickAt:

    def test_spec_parsing(self):
        assert WebAccess._parse_click_at_spec('412, 583') == (412.0, 583.0)
        assert WebAccess._parse_click_at_spec('12.5 40') == (12.5, 40.0)
        assert WebAccess._parse_click_at_spec('412') is None
        assert WebAccess._parse_click_at_spec('') is None

    def _wa(self):
        wa = make_wa()
        wa._browse_session_url = 'https://shop.example/kassa'
        wa._browse_pdf_text = None
        wa._tab = SimpleNamespace(send=AsyncMock(return_value=None))
        return wa

    def test_nothing_under_the_point_clicks_nothing(self):
        wa = self._wa()
        with patch.object(WebAccess, '_async_browse_ready_midsession', new=AsyncMock(return_value=None)), \
                patch.object(WebAccess, '_evaluate', new=AsyncMock(return_value=None)):
            result = run(wa._async_browse_click_at('9000, 9000'))
        assert result.startswith('ERROR: nothing is rendered at (9000, 9000)')
        wa._tab.send.assert_not_awaited()

    def test_dispatches_move_press_release_and_snapshots(self):
        wa = self._wa()
        with patch.object(WebAccess, '_async_browse_ready_midsession', new=AsyncMock(return_value=None)), \
                patch.object(WebAccess, '_evaluate',
                             new=AsyncMock(return_value='iframe src=https://js.stripe.com/v3/elements-inner-card-1.html')), \
                patch.object(WebAccess, '_POPUP_SETTLE_S', 0), \
                patch.object(WebAccess, '_async_wait_for_page_change', new=AsyncMock(return_value=('changed', None))), \
                patch.object(WebAccess, '_async_sync_popup_targets', new=AsyncMock()), \
                patch.object(WebAccess, '_async_wait_for_page_ready', new=AsyncMock()), \
                patch.object(WebAccess, '_async_take_snapshot', new=AsyncMock(return_value='SNAP')):
            result = run(wa._async_browse_click_at('412.5, 583'))
        assert result.startswith('Clicked at (412.5, 583) — hit '
                                 '<iframe src=https://js.stripe.com/v3/elements-inner-card-1.html>.\n\n')
        assert result.endswith('\n\nSNAP')
        sent = [c.args[0] for c in wa._tab.send.await_args_list]
        assert len(sent) == 3
        # each nodriver cdp call is a generator whose first yield is the wire message
        types = [next(g)['params']['type'] for g in sent]
        assert types == ['mouseMoved', 'mousePressed', 'mouseReleased']

    def _click(self, wa, hit, mark):
        with patch.object(WebAccess, '_async_browse_ready_midsession', new=AsyncMock(return_value=None)), \
                patch.object(WebAccess, '_evaluate', new=AsyncMock(return_value=hit)), \
                patch.object(WebAccess, '_async_action_mark', new=mark), \
                patch.object(WebAccess, '_async_dispatch_click_at', new=AsyncMock()), \
                patch.object(WebAccess, '_async_snapshot_after_action',
                             new=AsyncMock(return_value='SNAP')) as after:
            result = run(wa._async_browse_click_at('412, 583'))
        return result, after

    def test_click_into_iframe_skips_the_unchanged_verdict_and_says_so(self):
        # The main document's text signature cannot see what happened inside the
        # frame, so "the page looks exactly as before" would be noise — but a 3-D
        # Secure submit in a frame does drive the parent, so the page still gets its
        # time to react: the mark is taken, only the verdict is off.
        mark = AsyncMock(return_value={'sig': 'x'})
        result, after = self._click(self._wa(), 'iframe src=https://static-assets-pay.shop.example/v5', mark)
        mark.assert_awaited_once()
        assert after.await_args.args[0] == {'sig': 'x'}
        assert after.await_args.kwargs['verdict'] is False
        assert 'NOTE: the click landed inside an iframe' in result
        assert result.endswith('SNAP')

    def test_click_on_the_page_itself_keeps_the_change_detection(self):
        mark = AsyncMock(return_value={'sig': 'x'})
        result, after = self._click(self._wa(), 'button "Betala"', mark)
        mark.assert_awaited_once()
        assert after.await_args.args[0] == {'sig': 'x'}
        assert after.await_args.kwargs['verdict'] is True
        assert 'inside an iframe' not in result


@pytest.mark.unit
class TestSnapshotSkipsHiddenFrames:
    """A frame target whose hosting <iframe> is collapsed in the document that embeds
    it must not contribute elements: inside the frame its inputs look visible (the
    Radonova/Kustom checkout listed Stripe's 0px-tall autofill frame as card fields)."""

    KUSTOM = 'https://checkout.kustom.example/kcoc/checkout-template.html#order'
    STRIPE = 'https://js.stripe.example/v3/elements-inner-accessory-target-abc.html'

    def _snapshot(self, wa, main_extra, frame_targets, per_target):
        """per_target: target id -> (pre-pass children dict, collector result)."""
        async def fake_eval(_tab, expr):
            if 'document.title' in expr:
                return 'Checkout'
            return {'records': [{'tag': 'button', 'label': 'Main'}], **main_extra}

        async def frame_eval(_self, target_id, expr):
            children, collected = per_target[target_id]
            return children if 'shown' in expr and 'hidden' in expr else collected

        with patch.object(WebAccess, '_evaluate', side_effect=fake_eval), \
                patch.object(WebAccess, '_async_sync_popup_targets', new=AsyncMock()), \
                patch.object(WebAccess, '_async_list_frame_targets',
                             new=AsyncMock(return_value=frame_targets)), \
                patch.object(WebAccess, '_async_eval_in_frame_target', new=frame_eval):
            return run(wa._async_take_snapshot())

    def _wa(self):
        wa = make_wa()
        wa._tab = SimpleNamespace(target=FakeTargetInfo('main', 'page', 'https://shop.example/kassa'))
        wa._oopif_settle_giveups = frozenset()
        wa._blank_frame_target_ids = frozenset()
        return wa

    def test_frame_collapsed_inside_another_frame_is_left_out(self):
        targets = [FakeTargetInfo('stripe', url=self.STRIPE), FakeTargetInfo('kustom', url=self.KUSTOM)]
        per_target = {
            'kustom': ({'shown': [], 'hidden': [self.STRIPE]},
                       {'records': [{'tag': 'button', 'label': 'Betala köp', 'source': 'kustom'}]}),
            'stripe': ({'shown': [], 'hidden': []},
                       {'records': [{'tag': 'input', 'type': 'text', 'label': 'Kortnummer',
                                     'source': 'stripe'}]}),
        }
        result = self._snapshot(self._wa(), {'crossOriginFrames': [self.KUSTOM], 'hiddenFrames': []},
                                targets, per_target)

        assert 'Kortnummer' not in result
        assert 'Betala köp' in result
        # the kept frame's indices follow the main page's with no gap for the skipped one
        assert '[1] button "Betala köp"' in result

    def test_frame_collapsed_in_the_main_page_is_left_out(self):
        targets = [FakeTargetInfo('stripe', url=self.STRIPE)]
        per_target = {'stripe': ({'shown': [], 'hidden': []},
                                 {'records': [{'tag': 'input', 'type': 'text', 'label': 'Kortnummer',
                                               'source': 'stripe'}]})}
        result = self._snapshot(self._wa(), {'crossOriginFrames': [], 'hiddenFrames': [self.STRIPE]},
                                targets, per_target)

        assert 'Kortnummer' not in result
        assert 'Cross-origin iframes on this page' not in result

    def test_frame_shown_somewhere_survives_a_hidden_twin(self):
        # Stripe mounts several frames under one path: one collapsed, one the real card field.
        targets = [FakeTargetInfo('stripe', url=self.STRIPE), FakeTargetInfo('kustom', url=self.KUSTOM)]
        per_target = {
            'kustom': ({'shown': [self.STRIPE], 'hidden': [self.STRIPE]},
                       {'records': [{'tag': 'button', 'label': 'Betala köp', 'source': 'kustom'}]}),
            'stripe': ({'shown': [], 'hidden': []},
                       {'records': [{'tag': 'input', 'type': 'text', 'label': 'Kortnummer',
                                     'source': 'stripe'}]}),
        }
        result = self._snapshot(self._wa(), {'crossOriginFrames': [self.KUSTOM], 'hiddenFrames': []},
                                targets, per_target)

        assert 'Kortnummer' in result
        assert 'Betala köp' in result

    def test_twins_told_apart_by_fragment(self):
        # Stripe's express-pay bar renders (616x52) while its autofill twin stays 8px tall;
        # they share host and path and differ only in the fragment — the twin must still go.
        shown_twin = self.STRIPE + '#elementId=express'
        hidden_twin = self.STRIPE + '#elementId=autofill'
        targets = [FakeTargetInfo('bar', url=shown_twin), FakeTargetInfo('autofill', url=hidden_twin),
                   FakeTargetInfo('kustom', url=self.KUSTOM)]
        per_target = {
            'kustom': ({'shown': [{'src': shown_twin, 'w': 616, 'h': 52}],
                        'hidden': [{'src': hidden_twin, 'w': 300, 'h': 8}]},
                       {'records': [{'tag': 'button', 'label': 'Betala köp', 'source': 'kustom'}]}),
            'bar': ({'shown': [], 'hidden': []},
                    {'records': [{'tag': 'button', 'label': 'Köpa med GPay', 'source': 'bar'}]}),
            'autofill': ({'shown': [], 'hidden': []},
                         {'records': [{'tag': 'input', 'type': 'text', 'label': 'Kortnummer',
                                       'source': 'autofill'}]}),
        }
        result = self._snapshot(self._wa(), {'crossOriginFrames': [self.KUSTOM], 'hiddenFrames': []},
                                targets, per_target)

        assert 'Kortnummer' not in result
        assert 'Köpa med GPay' in result
        assert 'Betala köp' in result

    def test_frame_the_page_reports_covered_is_left_out(self):
        # The collector reports its own hidden iframes with a reason (Kustom's unselected
        # card form: display:block, 494x130, but in a 0px row with the next row painted
        # over it) — a dict entry hides the target the same as a bare src did.
        targets = [FakeTargetInfo('stripe', url=self.STRIPE)]
        per_target = {'stripe': ({'shown': [], 'hidden': []},
                                 {'records': [{'tag': 'input', 'type': 'text', 'label': 'Kortnummer',
                                               'source': 'stripe'}]})}
        result = self._snapshot(self._wa(),
                                {'crossOriginFrames': [],
                                 'hiddenFrames': [{'src': self.STRIPE, 'why': 'inside collapsed div, covered'}]},
                                targets, per_target)

        assert 'Kortnummer' not in result
