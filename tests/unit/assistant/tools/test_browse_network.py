"""browse_network: the per-session CDP Network capture (request → response → body / failure,
redirect hops, the ring buffer) and the report it renders. Motivated by the 2026-09-26 BankID
checkout that stalled with nothing to say why — the popup's own polling was invisible."""
import asyncio
import base64
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, Mock, patch
from nodriver import cdp

from assistant.tools.WebAccess import WebAccess


def bare_wa(session_url='https://shop.example/checkout'):
    wa = WebAccess.__new__(WebAccess)
    wa._browse_session_url = session_url
    wa._network_log = []
    wa._network_requests = {}
    return wa


def connection(target_id='popup-target-0001'):
    return SimpleNamespace(send=AsyncMock(return_value=('', False)),
                           target=SimpleNamespace(target_id=target_id))


def request_event(rid, url, method='GET', kind=cdp.network.ResourceType.XHR, post_data=None,
                  redirect_status=None):
    redirect = None
    if redirect_status is not None:
        redirect = SimpleNamespace(status=redirect_status, mime_type='text/html')
    return SimpleNamespace(request_id=cdp.network.RequestId(rid), type_=kind,
                           request=SimpleNamespace(url=url, method=method, post_data=post_data),
                           redirect_response=redirect)


def response_event(rid, status=200, mime='application/json'):
    return SimpleNamespace(request_id=cdp.network.RequestId(rid),
                           response=SimpleNamespace(status=status, mime_type=mime))


def finished_event(rid, size=512):
    return SimpleNamespace(request_id=cdp.network.RequestId(rid), encoded_data_length=size)


def failed_event(rid, error_text='net::ERR_CONNECTION_RESET', canceled=False):
    return SimpleNamespace(request_id=cdp.network.RequestId(rid), error_text=error_text, canceled=canceled)


def run(coro):
    return asyncio.run(coro)


def entry(wa, rid):
    return next(e for e in wa._network_log if e['rid'] == rid)


@pytest.mark.unit
class TestCapture:

    def test_only_api_and_document_traffic_is_recorded(self):
        wa = bare_wa()
        conn = connection()
        wa._on_network_request(request_event('img', 'https://shop.example/logo.png',
                                             kind=cdp.network.ResourceType.IMAGE), conn)
        wa._on_network_request(request_event('css', 'https://shop.example/app.css',
                                             kind=cdp.network.ResourceType.STYLESHEET), conn)
        wa._on_network_request(request_event('api', 'https://shop.example/api/order',
                                             kind=cdp.network.ResourceType.FETCH), conn)
        wa._on_network_request(request_event('doc', 'https://shop.example/thanks',
                                             kind=cdp.network.ResourceType.DOCUMENT), conn)
        # Late events for a dropped request must be ignored, not crash the handler.
        wa._on_network_response(response_event('img', mime='image/png'), conn)
        run(wa._on_network_finished(finished_event('img'), conn))
        wa._on_network_failed(failed_event('css'), conn)

        assert [e['rid'] for e in wa._network_log] == ['api', 'doc']
        assert conn.send.await_count == 0

    def test_form_post_body_is_url_decoded_so_the_vault_redactor_can_match_it(self):
        # The vault matches a password literally: percent-encoded it would slip past both the
        # log masker and the LLM-bound redaction, so the stored body is the decoded form.
        wa = bare_wa()
        with patch('assistant.tools.WebAccess.trace') as tr:
            wa._on_network_request(request_event(
                'login', 'https://shop.example/login', method='POST', kind=cdp.network.ResourceType.DOCUMENT,
                post_data='user=marcus%40example.com&answer=r%C3%A4tt+svar%26mer&remember=1'), connection())
        stored = entry(wa, 'login')['post']
        assert 'marcus@example.com' in stored and 'rätt svar&mer' in stored and 'remember=1' in stored
        assert '%40' not in stored
        traced = ' '.join(str(c.args[0]) for c in tr.call_args_list)
        assert '%40' not in traced and 'rätt svar' in traced

    @pytest.mark.parametrize("post, leaked, kept", [
        ('username=marcus&password=p%40ss+w0rd&otp=123456', ['p@ss w0rd', 'p%40ss', '123456'], ['marcus']),
        ('{"user":"marcus","password":"h\\u00e4mlig","card":{"cardNumber":"4111","pin":"9876"},'
         '"shipping":"home"}', ['hämlig', 'u00e4', '4111', '9876'], ['marcus', 'home']),
    ])
    def test_credential_fields_are_masked_by_name_before_the_body_is_stored_or_traced(self, post, leaked, kept):
        wa = bare_wa()
        with patch('assistant.tools.WebAccess.trace') as tr:
            wa._on_network_request(request_event('login', 'https://shop.example/api/login', method='POST',
                                                 post_data=post), connection())
        stored = entry(wa, 'login')['post']
        traced = ' '.join(str(c.args[0]) for c in tr.call_args_list)
        for value in leaked:
            assert value not in stored and value not in traced
        for value in kept:
            assert value in stored
        report = run(wa._async_browse_network(''))
        for value in leaked:
            assert value not in report

    def test_bodies_that_are_neither_form_nor_json_are_kept_verbatim(self):
        wa = bare_wa()
        wa._on_network_request(request_event('q', 'https://shop.example/graphql', method='POST',
                                             post_data='query { order(id: "KEP47L") { status } }'), connection())
        assert entry(wa, 'q')['post'] == 'query { order(id: "KEP47L") { status } }'

    def test_long_form_like_body_that_is_not_a_form_is_classified_without_backtracking(self):
        # Many `key=value&` fields and then whitespace: the shape that pinned the server's GIL for
        # hours when the form check could split each field several ways. Kept verbatim, instantly.
        post = '&'.join(f'field{i}=value{i}' for i in range(60)) + '&note=two words'
        wa = bare_wa()
        wa._on_network_request(request_event('beacon', 'https://news.example/ingest', method='POST',
                                             post_data=post), connection())
        assert entry(wa, 'beacon')['post'] == post

    def test_json_xhr_body_is_read_once_loading_finishes(self):
        wa = bare_wa()
        conn = connection()
        conn.send.return_value = ('{"status": "pending", "orderId": "KEP47L"}', False)
        wa._on_network_request(request_event('poll', 'https://pay.example/api/poll', method='POST',
                                             post_data='{"order": "KEP47L"}'), conn)
        wa._on_network_response(response_event('poll', status=200, mime='application/json'), conn)
        run(wa._on_network_finished(finished_event('poll', size=48), conn))

        e = entry(wa, 'poll')
        assert (e['status'], e['size'], e['done']) == (200, 48, True)
        assert e['post'] == '{"order": "KEP47L"}'
        assert '"pending"' in e['body']
        body_request = conn.send.await_args.args[0].send(None)
        assert body_request['method'] == 'Network.getResponseBody'
        assert body_request['params']['requestId'] == 'poll'

    def test_base64_body_is_decoded(self):
        wa = bare_wa()
        conn = connection()
        conn.send.return_value = (base64.b64encode('{"state":"complete"}'.encode()).decode(), True)
        wa._on_network_request(request_event('r', 'https://pay.example/api/state'), conn)
        wa._on_network_response(response_event('r'), conn)
        run(wa._on_network_finished(finished_event('r'), conn))
        assert entry(wa, 'r')['body'] == '{"state":"complete"}'

    @pytest.mark.parametrize("kind, mime", [
        (cdp.network.ResourceType.DOCUMENT, 'text/html'),      # a page load: too big, not what we want
        (cdp.network.ResourceType.XHR, 'image/png'),           # binary XHR
        (cdp.network.ResourceType.FETCH, 'application/octet-stream'),
    ])
    def test_bodies_are_not_fetched_for_pages_or_binary_responses(self, kind, mime):
        wa = bare_wa()
        conn = connection()
        wa._on_network_request(request_event('r', 'https://shop.example/x', kind=kind), conn)
        wa._on_network_response(response_event('r', mime=mime), conn)
        run(wa._on_network_finished(finished_event('r'), conn))
        assert conn.send.await_count == 0
        assert entry(wa, 'r')['body'] is None
        assert entry(wa, 'r')['done'] is True

    def test_unavailable_body_leaves_the_entry_intact(self):
        # Chrome evicts bodies quickly and a popup may already be gone: the
        # status/size already recorded must survive the failed body fetch.
        wa = bare_wa()
        conn = connection()
        conn.send.side_effect = RuntimeError("No resource with given identifier found")
        wa._on_network_request(request_event('r', 'https://pay.example/api/poll'), conn)
        wa._on_network_response(response_event('r', status=200), conn)
        run(wa._on_network_finished(finished_event('r', size=10), conn))
        e = entry(wa, 'r')
        assert e['body'] is None
        assert (e['status'], e['size'], e['done']) == (200, 10, True)

    def test_body_read_is_bounded_by_its_own_timeout(self):
        wa = bare_wa()
        conn = connection()
        conn.send.side_effect = lambda *a, **k: asyncio.sleep(30)
        wa._on_network_request(request_event('r', 'https://pay.example/api/poll'), conn)
        wa._on_network_response(response_event('r'), conn)
        with patch.object(WebAccess, '_NETWORK_BODY_FETCH_TIMEOUT', 0.02):
            run(wa._on_network_finished(finished_event('r'), conn))
        assert entry(wa, 'r')['body'] is None

    def test_redirect_closes_the_previous_hop_and_continues_under_the_same_id(self):
        wa = bare_wa()
        conn = connection()
        wa._on_network_request(request_event('nav', 'https://pay.example/return',
                                             kind=cdp.network.ResourceType.DOCUMENT), conn)
        wa._on_network_request(request_event('nav', 'https://shop.example/thanks',
                                             kind=cdp.network.ResourceType.DOCUMENT, redirect_status=302), conn)
        wa._on_network_response(response_event('nav', status=200, mime='text/html'), conn)

        first, second = wa._network_log
        assert first['url'] == 'https://pay.example/return'
        assert (first['status'], first['done']) == (302, True)
        assert 'shop.example/thanks' in first['note']
        assert second['url'] == 'https://shop.example/thanks'
        assert second['status'] == 200

    def test_failure_and_cancellation_are_noted(self):
        wa = bare_wa()
        conn = connection()
        wa._on_network_request(request_event('a', 'https://pay.example/api/a'), conn)
        wa._on_network_request(request_event('b', 'https://pay.example/api/b'), conn)
        wa._on_network_failed(failed_event('a', error_text='net::ERR_CONNECTION_RESET'), conn)
        wa._on_network_failed(failed_event('b', canceled=True), conn)
        assert 'ERR_CONNECTION_RESET' in entry(wa, 'a')['note']
        assert entry(wa, 'b')['note'] == 'canceled'
        assert entry(wa, 'a')['done'] and entry(wa, 'b')['done']

    def test_entries_are_tagged_with_the_tab_that_made_them(self):
        wa = bare_wa()
        wa._on_network_request(request_event('base', 'https://shop.example/api/cart'),
                               connection('AAAAAAAA-base-target'))
        wa._on_network_request(request_event('pop', 'https://pay.example/api/poll'),
                               connection('BBBBBBBB-popup-target'))
        assert entry(wa, 'base')['tab'] != entry(wa, 'pop')['tab']
        assert entry(wa, 'pop')['tab'] in 'BBBBBBBB-popup-target'

    def test_ring_buffer_drops_the_oldest_but_keeps_a_reused_request_id(self):
        wa = bare_wa()
        conn = connection()
        with patch.object(WebAccess, '_NETWORK_LOG_MAX', 2):
            wa._on_network_request(request_event('nav', 'https://pay.example/hop1',
                                                 kind=cdp.network.ResourceType.DOCUMENT), conn)
            wa._on_network_request(request_event('nav', 'https://pay.example/hop2',
                                                 kind=cdp.network.ResourceType.DOCUMENT, redirect_status=301), conn)
            wa._on_network_request(request_event('x', 'https://pay.example/api/x'), conn)
        assert [e['url'] for e in wa._network_log] == ['https://pay.example/hop2', 'https://pay.example/api/x']
        # The evicted hop1 shared its id with hop2, which is still live: its response must land.
        wa._on_network_response(response_event('nav', status=200, mime='text/html'), conn)
        assert entry(wa, 'nav')['status'] == 200

    def test_late_events_for_an_evicted_request_are_ignored(self):
        wa = bare_wa()
        conn = connection()
        with patch.object(WebAccess, '_NETWORK_LOG_MAX', 1):
            wa._on_network_request(request_event('old', 'https://pay.example/api/old'), conn)
            wa._on_network_request(request_event('new', 'https://pay.example/api/new'), conn)
        wa._on_network_response(response_event('old', status=500), conn)
        assert [e['rid'] for e in wa._network_log] == ['new']
        assert 'old' not in wa._network_requests

    def test_installation_survives_a_registration_failure(self):
        wa = bare_wa()
        tab = SimpleNamespace(add_handler=Mock(side_effect=RuntimeError("no connection")))
        wa._install_network_capture(tab)  # must not raise


@pytest.mark.unit
class TestReport:

    def captured(self, wa, rid, url, status=200, body=None, post=None, kind=cdp.network.ResourceType.XHR):
        conn = connection()
        wa._on_network_request(request_event(rid, url, method='POST' if post else 'GET', kind=kind,
                                             post_data=post), conn)
        if status is not None:
            wa._on_network_response(response_event(rid, status=status), conn)
        if body is not None:
            conn.send.return_value = (body, False)
            run(wa._on_network_finished(finished_event(rid), conn))

    def test_no_session_is_an_error(self):
        wa = bare_wa(session_url=None)
        assert run(wa._async_browse_network('')).startswith('ERROR')

    def test_empty_session_says_so(self):
        wa = bare_wa()
        report = run(wa._async_browse_network(''))
        assert not report.startswith('ERROR')
        assert 'browse_open' in report

    def test_the_listing_is_what_the_history_filter_drops_filtered_or_not(self):
        """The memory curator and search_history know a network listing by its opening words;
        the report and the filter must agree, or the listings fill the transcript and the hits
        again. The whole session and a url-filtered part of it are both listings."""
        from assistant.commands.Commands import is_reference_result
        wa = bare_wa()
        self.captured(wa, 'first', 'https://pay.example/api/start', body='{"status": "ok"}')
        self.captured(wa, 'second', 'https://pay.example/api/poll', body='{"status": "pending"}')
        assert is_reference_result(run(wa._async_browse_network('')))
        assert is_reference_result(run(wa._async_browse_network('api/poll')))

    def test_entries_are_listed_oldest_first_with_bodies(self):
        wa = bare_wa()
        self.captured(wa, 'first', 'https://pay.example/api/start', post='{"order":\n  "KEP47L"}')
        self.captured(wa, 'second', 'https://pay.example/api/poll', body='{\n  "status": "pending"\n}')
        self.captured(wa, 'third', 'https://pay.example/api/poll', body='{"status": "complete"}')
        report = run(wa._async_browse_network(''))
        assert report.index('api/start') < report.index('"pending"') < report.index('"complete"')
        assert '> {"order": "KEP47L"}' in report      # request body, whitespace collapsed
        assert '< { "status": "pending" }' in report  # response body, whitespace collapsed

    def test_unanswered_and_failed_requests_are_marked(self):
        wa = bare_wa()
        self.captured(wa, 'hang', 'https://pay.example/api/hang', status=None)
        conn = connection()
        wa._on_network_request(request_event('dead', 'https://pay.example/api/dead'), conn)
        wa._on_network_failed(failed_event('dead', error_text='net::ERR_ABORTED'), conn)
        report = run(wa._async_browse_network(''))
        hang_line = next(l for l in report.splitlines() if 'api/hang' in l)
        dead_line = next(l for l in report.splitlines() if 'api/dead' in l)
        assert '…' in hang_line
        assert 'ERR_ABORTED' in dead_line

    def test_url_filter_is_a_regex_that_keeps_session_indices(self):
        wa = bare_wa()
        self.captured(wa, 'a', 'https://shop.example/api/cart')
        self.captured(wa, 'b', 'https://pay.example/api/poll')
        self.captured(wa, 'c', 'https://pay.example/api/status')
        report = run(wa._async_browse_network('pay\\.example/api/(poll|status)'))
        assert 'api/cart' not in report
        assert '[1] ' in report and '[2] ' in report and '[0] ' not in report

    def test_invalid_filter_regex_matches_literally(self):
        wa = bare_wa()
        self.captured(wa, 'a', 'https://pay.example/api/poll?q=(x')
        self.captured(wa, 'b', 'https://pay.example/api/other')
        report = run(wa._async_browse_network('poll?q=(x'))
        assert 'api/poll' in report and 'api/other' not in report

    def test_filter_without_hits_reports_the_session_total(self):
        wa = bare_wa()
        self.captured(wa, 'a', 'https://shop.example/api/cart')
        report = run(wa._async_browse_network('bankid'))
        assert 'bankid' in report and '1 entries total' in report

    def test_only_the_newest_entries_are_shown_when_over_the_cap(self):
        wa = bare_wa()
        for i in range(5):
            self.captured(wa, f'r{i}', f'https://pay.example/api/poll/{i}')
        with patch.object(WebAccess, '_NETWORK_REPORT_DEFAULT', 2):
            report = run(wa._async_browse_network(''))
        assert 'api/poll/3' in report and 'api/poll/4' in report
        assert 'api/poll/2' not in report
        assert '2 of 5' in report

    def test_oversized_report_is_cut_from_the_old_end(self):
        wa = bare_wa()
        self.captured(wa, 'old', 'https://pay.example/api/old', body='x' * 200)
        self.captured(wa, 'new', 'https://pay.example/api/new', body='y' * 200)
        with patch.object(WebAccess, '_NETWORK_REPORT_MAX_CHARS', 250):
            report = run(wa._async_browse_network(''))
        assert 'older entries cut' in report
        assert 'api/new' in report and 'api/old' not in report


@pytest.mark.unit
class TestSessionLifecycle:

    def test_closing_the_browser_drops_the_capture(self):
        wa = bare_wa()
        wa._browse_pdf_text = None
        wa._browse_fail_streak = 3
        wa._browser_hard_down = False
        wa._browser_closed_by_user = False
        wa._last_init_failure_ts = 0.0
        wa._on_network_request(request_event('r', 'https://pay.example/api/poll'), connection())
        wa._finish_browse_close()
        assert wa._network_log == [] and wa._network_requests == {}
        assert run(wa._async_browse_network('')).startswith('ERROR')

    def test_adopting_a_popup_captures_its_traffic_too(self):
        # The payment popup's own polling is the point of the feature — a capture
        # on the base tab alone would miss it.
        wa = bare_wa()
        wa._tab = SimpleNamespace(name='base')
        wa._popup_parents = []
        wa._pending_popup_targets = {}
        wa._pending_target_notes = []
        wa._browse_frame_sessions = {}
        popup = SimpleNamespace(add_handler=Mock(), send=AsyncMock())
        with patch.object(WebAccess, '_registry_target', return_value=object()), \
                patch.object(WebAccess, '_as_tab', return_value=popup), \
                patch.object(WebAccess, '_async_wait_for_page_ready', new=AsyncMock()):
            run(wa._async_adopt_popup_target('popup-target-0001', 'https://pay.example/bankid'))
        listened = {call.args[0] for call in popup.add_handler.call_args_list}
        assert cdp.network.RequestWillBeSent in listened
        assert cdp.network.LoadingFinished in listened
        assert wa._tab is popup
