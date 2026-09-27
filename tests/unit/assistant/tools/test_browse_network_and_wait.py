"""browse_network and browse_wait (2026-09-26): the CDP network capture that tells a
rejected BankID signing from one that never completed, and the polling wait that
replaces hand-rolled sleeps inside browse_eval (which tripped the 10 s eval timeout)."""
import asyncio
import base64
import time
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, Mock, patch
from nodriver import cdp

from assistant.tools.WebAccess import WebAccess

XHR = cdp.network.ResourceType.XHR


def make_wa(session='https://shop.example/kassa'):
    wa = WebAccess.__new__(WebAccess)
    wa._browse_session_url = session
    wa._browse_pdf_text = None
    wa._tab = SimpleNamespace()
    wa._dialog_notes = []
    wa._network_log = []
    wa._network_requests = {}
    return wa


def run(coro):
    return asyncio.run(coro)


class FakeConnection:
    """A tab's CDP connection: answers Network.getResponseBody from a table of
    request id -> (body, base64Encoded) and records which bodies were asked for."""

    def __init__(self, tab_id='popup-1234567', bodies=None, error=None):
        self.target = SimpleNamespace(target_id=tab_id)
        self.bodies = bodies or {}
        self.error = error
        self.body_requests = []

    async def send(self, gen, **kwargs):
        rid = str(gen.send(None)['params']['requestId'])
        self.body_requests.append(rid)
        if self.error is not None:
            raise self.error
        return self.bodies[rid]


def request_event(rid, url, *, method='GET', type_=XHR, post=None, redirect=None):
    return SimpleNamespace(request_id=cdp.network.RequestId(rid), type_=type_, redirect_response=redirect,
                           request=SimpleNamespace(url=url, method=method, post_data=post))


def response_event(rid, status=200, mime='application/json'):
    return SimpleNamespace(request_id=cdp.network.RequestId(rid),
                           response=SimpleNamespace(status=status, mime_type=mime))


def finished_event(rid, size=512):
    return SimpleNamespace(request_id=cdp.network.RequestId(rid), encoded_data_length=size)


def failed_event(rid, error_text='net::ERR_CONNECTION_RESET', canceled=False):
    return SimpleNamespace(request_id=cdp.network.RequestId(rid), error_text=error_text, canceled=canceled)


def exchange(wa, conn, rid, url, *, type_=XHR, method='GET', status=200,
             mime='application/json', post=None, size=512):
    """Drive one request through the three CDP events the capture listens to."""
    wa._on_network_request(request_event(rid, url, method=method, type_=type_, post=post), conn)
    wa._on_network_response(response_event(rid, status, mime), conn)
    run(wa._on_network_finished(finished_event(rid, size), conn))
    return wa._network_requests[rid]


@pytest.mark.unit
class TestNetworkCapture:

    def test_only_api_and_document_traffic_is_recorded(self):
        wa = make_wa()
        conn = FakeConnection()
        for rid, url, kind in [('1', 'https://cdn.example/logo.png', cdp.network.ResourceType.IMAGE),
                               ('2', 'https://cdn.example/app.js', cdp.network.ResourceType.SCRIPT),
                               ('3', 'https://cdn.example/app.css', cdp.network.ResourceType.STYLESHEET),
                               ('4', 'https://api.example/order/status', 'XHR'),
                               ('5', 'https://shop.example/callback', 'Document'),
                               ('6', 'https://api.example/session', cdp.network.ResourceType.FETCH)]:
            wa._on_network_request(request_event(rid, url, type_=kind), conn)
        assert [e['url'] for e in wa._network_log] == [
            'https://api.example/order/status', 'https://shop.example/callback', 'https://api.example/session']

    def test_json_xhr_body_is_read_when_loading_finishes(self):
        wa = make_wa()
        conn = FakeConnection(tab_id='basetab-9999', bodies={'7': ('{"status": "pending"}', False)})
        entry = exchange(wa, conn, '7', 'https://api.example/order/status', post='{"id": 12}', size=99)
        assert (entry['status'], entry['mime'], entry['size']) == (200, 'application/json', 99)
        assert entry['post'] == '{"id": 12}'
        assert entry['body'] == '{"status": "pending"}'
        assert entry['done'] is True
        assert entry['tab'] == 'basetab-'  # the tab id is shortened to keep report lines readable

    def test_documents_and_binary_responses_do_not_fetch_a_body(self):
        wa = make_wa()
        conn = FakeConnection(bodies={'1': ('<html>', False), '2': ('...', True)})
        page = exchange(wa, conn, '1', 'https://shop.example/callback', type_='Document', mime='text/html')
        blob = exchange(wa, conn, '2', 'https://api.example/receipt', mime='application/pdf')
        assert conn.body_requests == []
        assert page['body'] is None and blob['body'] is None
        assert page['done'] and blob['done']

    def test_base64_body_is_decoded(self):
        wa = make_wa()
        encoded = base64.b64encode('{"signed": true}'.encode()).decode()
        conn = FakeConnection(bodies={'1': (encoded, True)})
        entry = exchange(wa, conn, '1', 'https://api.example/bankid/collect', mime='text/plain')
        assert entry['body'] == '{"signed": true}'

    def test_unavailable_body_leaves_the_entry_finished_without_one(self):
        wa = make_wa()
        conn = FakeConnection(error=RuntimeError('No resource with given identifier found'))
        entry = exchange(wa, conn, '1', 'https://api.example/order/status')
        assert entry['body'] is None
        assert entry['done'] is True
        assert entry['status'] == 200  # the response itself was still recorded

    def test_redirect_closes_the_first_hop_and_continues_on_the_new_url(self):
        wa = make_wa()
        conn = FakeConnection()
        wa._on_network_request(request_event('1', 'https://shop.example/pay', type_='Document'), conn)
        hop = SimpleNamespace(status=302, mime_type='')
        wa._on_network_request(
            request_event('1', 'https://bank.example/login', type_='Document', redirect=hop), conn)
        wa._on_network_response(response_event('1', 200, 'text/html'), conn)

        first, second = wa._network_log
        assert first['status'] == 302 and first['done'] is True
        assert 'https://bank.example/login' in first['note']
        assert second['url'] == 'https://bank.example/login'
        assert second['status'] == 200  # later events for the id belong to the newest hop

    def test_failed_and_canceled_requests_are_noted(self):
        wa = make_wa()
        conn = FakeConnection()
        wa._on_network_request(request_event('1', 'https://api.example/a'), conn)
        wa._on_network_request(request_event('2', 'https://api.example/b'), conn)
        wa._on_network_failed(failed_event('1'), conn)
        wa._on_network_failed(failed_event('2', canceled=True), conn)
        failed, canceled = wa._network_log
        assert failed['done'] and 'net::ERR_CONNECTION_RESET' in failed['note']
        assert canceled['done'] and canceled['note'] == 'canceled'

    def test_ring_keeps_the_newest_entries_and_forgets_evicted_ids(self):
        wa = make_wa()
        conn = FakeConnection()
        with patch.object(WebAccess, '_NETWORK_LOG_MAX', 3):
            for i in range(5):
                wa._on_network_request(request_event(str(i), f'https://api.example/poll/{i}'), conn)
            wa._on_network_response(response_event('0', 500), conn)  # late reply for an evicted id
            wa._on_network_response(response_event('4', 200), conn)
        assert [e['url'] for e in wa._network_log] == [f'https://api.example/poll/{i}' for i in (2, 3, 4)]
        assert set(wa._network_requests) == {'2', '3', '4'}
        assert all(e['status'] != 500 for e in wa._network_log)
        assert wa._network_log[-1]['status'] == 200

    def test_events_for_requests_never_seen_are_ignored(self):
        wa = make_wa()
        conn = FakeConnection()
        wa._on_network_response(response_event('9'), conn)
        run(wa._on_network_finished(finished_event('9'), conn))
        wa._on_network_failed(failed_event('9'), conn)
        assert wa._network_log == []
        assert conn.body_requests == []

    def test_closing_the_session_drops_the_capture(self):
        wa = make_wa()
        conn = FakeConnection()
        exchange(wa, conn, '1', 'https://api.example/order/status', mime='text/plain')
        wa._browse_fail_streak = 2
        wa._browser_hard_down = False
        wa._browser_closed_by_user = False
        wa._last_init_failure_ts = 0.0
        wa._finish_browse_close()
        wa._on_network_response(response_event('1', 500), conn)  # straggler from the dead session
        assert wa._network_log == []
        assert wa._network_requests == {}


class FakeTargetInfo:
    def __init__(self, target_id, url):
        self.target_id = cdp.target.TargetID(target_id)
        self.type_ = 'page'
        self.url = url


class FakeTab:
    """A nodriver page target that records handler registrations."""

    def __init__(self, target_id, url):
        self.target = FakeTargetInfo(target_id, url)
        self.handlers = {}

    async def evaluate(self, *args, **kwargs):  # satisfies _as_tab's Tab check
        return None

    def add_handler(self, event_cls, handler):
        self.handlers[event_cls] = handler


@pytest.mark.unit
class TestPopupCapture:

    def test_traffic_on_an_adopted_popup_is_captured_and_attributed_to_it(self):
        base = FakeTab('base0000-1111', 'https://shop.example/kassa')
        popup = FakeTab('pop00000-2222', 'https://bank.example/bankid')
        wa = make_wa()
        wa._tab = base
        wa._browser = SimpleNamespace(targets=[base, popup])
        wa._popup_parents = []
        wa._browse_frame_sessions = {}
        wa._pending_target_notes = []
        wa._pending_popup_targets = {}
        with patch.object(WebAccess, '_async_apply_anti_bot_tweaks', new=AsyncMock()), \
                patch.object(WebAccess, '_async_wait_for_page_ready', new=AsyncMock()):
            run(wa._async_adopt_popup_target('pop00000-2222', 'https://bank.example/bankid'))
        assert wa._tab is popup

        on_request = popup.handlers[cdp.network.RequestWillBeSent]
        on_request(request_event('1', 'https://bank.example/collect', method='POST'), popup)
        assert [(e['tab'], e['url']) for e in wa._network_log] == [('pop00000', 'https://bank.example/collect')]


@pytest.mark.unit
class TestBrowseNetworkReport:

    def test_needs_a_session(self):
        wa = make_wa(session=None)
        assert run(wa._async_browse_network('')).startswith('ERROR')

    def test_report_is_oldest_first_with_bodies_collapsed_to_one_line(self):
        wa = make_wa()
        conn = FakeConnection(bodies={'1': ('{\n  "status": "pending"\n}', False), '2': ('{"n": 3}', False)})
        exchange(wa, conn, '1', 'https://api.example/order/status', method='POST', post='{"id":\n 12}')
        wa._on_network_request(request_event('0', 'https://cdn.example/logo.png',
                                             type_=cdp.network.ResourceType.IMAGE), conn)
        exchange(wa, conn, '2', 'https://api.example/cart', size=0)

        report = run(wa._async_browse_network(''))
        assert report.index('https://api.example/order/status') < report.index('https://api.example/cart')
        assert 'logo.png' not in report
        assert '> {"id": 12}' in report
        assert '< { "status": "pending" }' in report
        assert 'POST 200 https://api.example/order/status' in report

    def test_url_filter_is_a_case_insensitive_regex(self):
        wa = make_wa()
        conn = FakeConnection()
        exchange(wa, conn, '1', 'https://pay.klarna.example/Order/status', mime='text/plain')
        exchange(wa, conn, '2', 'https://shop.example/cart', mime='text/plain')
        report = run(wa._async_browse_network('klarna.*order|nothing-else'))
        assert 'Order/status' in report
        assert 'shop.example/cart' not in report
        assert '1 of 1 matching entries (2 captured' in report

    def test_invalid_regex_filter_falls_back_to_a_literal_match(self):
        wa = make_wa()
        conn = FakeConnection()
        exchange(wa, conn, '1', 'https://api.example/poll(1)', mime='text/plain')
        exchange(wa, conn, '2', 'https://api.example/poll', mime='text/plain')
        report = run(wa._async_browse_network('poll(1'))
        assert 'poll(1)' in report
        assert report.count('https://api.example/poll') == 1

    def test_nothing_matching_says_so_and_gives_the_session_total(self):
        wa = make_wa()
        conn = FakeConnection()
        exchange(wa, conn, '1', 'https://api.example/cart', mime='text/plain')
        exchange(wa, conn, '2', 'https://api.example/session', mime='text/plain')
        report = run(wa._async_browse_network('bankid'))
        assert not report.startswith('Network log')
        assert 'bankid' in report and '2 entries total' in report
        assert 'total' not in run(make_wa()._async_browse_network(''))

    def test_long_sessions_report_only_the_newest_entries(self):
        wa = make_wa()
        conn = FakeConnection()
        for i in range(5):
            exchange(wa, conn, str(i), f'https://api.example/poll/{i}', mime='text/plain')
        with patch.object(WebAccess, '_NETWORK_REPORT_DEFAULT', 3):
            report = run(wa._async_browse_network(''))
        assert '3 of 5 matching entries (5 captured' in report
        assert 'poll/1 ' not in report and 'poll/2 ' in report and 'poll/4 ' in report


@pytest.mark.unit
class TestParseWaitSpec:

    def test_trailing_number_is_the_timeout(self):
        assert WebAccess._parse_wait_spec('Continue|callback, 45') == ('Continue|callback', 45.0)

    def test_a_comma_inside_the_pattern_survives(self):
        pattern, timeout = WebAccess._parse_wait_spec('Tack, välkommen')
        assert pattern == 'Tack, välkommen'
        assert timeout == WebAccess._WAIT_DEFAULT_S
        assert WebAccess._parse_wait_spec('order, complete, 20') == ('order, complete', 20.0)

    def test_quotes_are_stripped_and_fractions_accepted(self):
        assert WebAccess._parse_wait_spec('"order (pending|complete)", 12.5') == ('order (pending|complete)', 12.5)

    def test_empty_spec_has_no_pattern(self):
        assert WebAccess._parse_wait_spec('')[0] == ''


class FakeClock:
    """time.time() that jumps a fixed step per call so a wait's deadline passes without sleeping."""

    def __init__(self, step):
        self.now = 1_000_000.0
        self.step = step
        self.strftime, self.localtime = time.strftime, time.localtime

    def time(self):
        self.now += self.step
        return self.now


def state(url='https://bank.example/login', title='Log in', text='Open BankID on this device'):
    return {'url': url, 'title': title, 'text': text}


@pytest.mark.unit
class TestBrowseWait:

    def _patches(self, probes):
        return [patch.object(WebAccess, '_async_browse_ready_midsession', new=AsyncMock(return_value=None)),
                patch.object(WebAccess, '_async_sync_popup_targets', new=AsyncMock()),
                patch.object(WebAccess, '_async_wait_for_page_ready', new=AsyncMock()),
                patch.object(WebAccess, '_async_wait_for_settle', new=AsyncMock()),
                patch.object(WebAccess, '_async_take_snapshot', new=AsyncMock(return_value='SNAP')),
                patch.object(WebAccess, '_evaluate', new=AsyncMock(side_effect=probes)),
                patch.object(WebAccess, '_WAIT_POLL_S', 0.001)]

    def _wait(self, wa, pattern, timeout=30.0, probes=()):
        patches = self._patches(list(probes))
        for p in patches:
            p.start()
        try:
            result = run(wa._async_browse_wait(pattern, timeout))
            return result, {p.attribute: p.new for p in patches}
        finally:
            for p in patches:
                p.stop()

    def test_follows_popups_each_poll_and_snapshots_once_the_url_matches(self):
        wa = make_wa()
        result, mocks = self._wait(wa, 'callback|bekraftelse', probes=[
            state(), state(), state(url='https://shop.example/callback?order=17')])
        assert result.startswith('MATCHED')
        assert result.endswith('SNAP')
        assert mocks['_async_sync_popup_targets'].await_count == 3
        mocks['_async_wait_for_page_ready'].assert_awaited_once()

    def test_matches_visible_text_regardless_of_case(self):
        wa = make_wa()
        result, _ = self._wait(wa, 'tack för din', probes=[state(text='Tack för din beställning!')])
        assert result.startswith('MATCHED')

    def test_invalid_regex_is_matched_literally(self):
        wa = make_wa()
        result, _ = self._wait(wa, 'Order (', probes=[state(text='Order (pending)')])
        assert result.startswith('MATCHED')

    def test_a_failed_probe_is_retried_instead_of_ending_the_wait(self):
        wa = make_wa()
        result, mocks = self._wait(wa, 'signed', probes=[asyncio.TimeoutError(), state(text='Signed!')])
        assert result.startswith('MATCHED')
        assert mocks['_evaluate'].await_count == 2

    def test_expiry_returns_the_current_page_and_points_at_browse_network(self):
        wa = make_wa()
        clock = FakeClock(step=0.4)
        with patch('assistant.tools.WebAccess.time', clock):
            result, mocks = self._wait(wa, 'bekraftelse', timeout=1.0, probes=[state()] * 20)
        assert result.startswith('TIMEOUT')
        assert 'browse_network' in result
        assert result.endswith('SNAP')
        assert mocks['_evaluate'].await_count > 1  # kept polling until the deadline, not one look
        mocks['_async_wait_for_page_ready'].assert_not_awaited()

    def test_empty_pattern_is_refused_without_touching_the_page(self):
        wa = make_wa()
        result, mocks = self._wait(wa, '', probes=[state()])
        assert result.startswith('ERROR')
        mocks['_evaluate'].assert_not_awaited()

    def test_dialog_notes_lead_the_reply(self):
        wa = make_wa()
        wa._dialog_notes = ['NOTE: confirm dialog accepted']
        result, _ = self._wait(wa, 'login', probes=[state()])
        assert result.startswith('NOTE: confirm dialog accepted')
        assert 'MATCHED' in result

    def test_sync_command_caps_the_wait_and_widens_the_watchdog_beyond_it(self):
        wa = make_wa()
        wa._run_browse_op = Mock(return_value='done')
        assert wa.browse_wait('callback, 500') == 'done'
        _, coro, _ = wa._run_browse_op.call_args.args
        try:
            assert coro.cr_frame.f_locals['timeout_s'] == WebAccess._WAIT_MAX_S
        finally:
            coro.close()
        assert wa._run_browse_op.call_args.kwargs['timeout'] > WebAccess._WAIT_MAX_S
