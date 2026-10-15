"""The network stage of the readiness gate and the blank-page re-read of fetch_via_chrome.

Motivated by the 2026-10-15 Apple order page: a static shell whose DOM went quiet before
its fetchOrder XHR was even sent, so open_url grabbed zero visible text and fell back to
summarising the analytics JSON while the order arrived 0.4 s later."""
import asyncio
import threading
import time

import pytest
from unittest.mock import AsyncMock, Mock, patch

from assistant.tools.WebAccess import WebAccess


def run(coro):
    return asyncio.run(coro)


def make_wa():
    wa = WebAccess.__new__(WebAccess)
    wa._PAGE_READY_POLL = 0.001
    wa._PAGE_NET_IDLE_MS = 5
    wa._PAGE_NET_IDLE_TIMEOUT = 0.05
    wa._BLANK_REGRAB_WAIT_S = 0.001
    wa._browser = Mock()
    wa._tab = Mock()
    wa._driver_ready = threading.Event()
    wa._driver_ready.set()
    wa._clear_network_log()
    return wa


def pending(wa, rid, url='https://shop.example/api/order', kind='XHR', age=0.0):
    entry = {'rid': rid, 'ts': time.time() - age, 'tab': 't', 'type': kind, 'method': 'POST',
             'url': url, 'post': '', 'status': None, 'mime': '', 'size': None, 'body': None,
             'note': '', 'done': False}
    wa._network_add(entry)
    return entry


SHELL = """<html><body>
<div id="spinner" class="loading"></div>
<script type="application/json" id="omniture">{"page":{"name":"AOS: order status","locale":"sv_SE"}}</script>
</body></html>"""

RENDERED = """<html><body><main><h1>Order W1608015212</h1>
<p>Shipped — arrives Friday 17 October.</p></main></body></html>"""


@pytest.mark.unit
class TestWaitForNetworkIdle:

    def test_a_page_without_captured_traffic_passes_straight_through(self):
        wa = make_wa()
        started = time.time()
        assert run(wa._async_wait_for_network_idle()) == "idle:no-traffic"
        assert time.time() - started < 0.02

    def test_waits_for_an_in_flight_request_to_finish(self):
        wa = make_wa()
        entry = pending(wa, 'r1')

        async def scenario():
            async def finish_later():
                await asyncio.sleep(0.01)
                entry['done'] = True
                wa._network_touch()
            asyncio.ensure_future(finish_later())
            return await wa._async_wait_for_network_idle()

        assert run(scenario()) == "idle"
        assert entry['done']

    def test_a_request_sent_right_after_dom_quiet_is_waited_for(self):
        """The Apple case: nothing in flight when the DOM went quiet, the data fetch
        sent a moment later — the idle period must restart on every new request."""
        wa = make_wa()
        wa._PAGE_NET_IDLE_MS = 20
        wa._PAGE_NET_IDLE_TIMEOUT = 0.5
        wa._network_touch()
        order = []

        async def scenario():
            async def late_fetch():
                await asyncio.sleep(0.005)
                entry = pending(wa, 'fetchOrder')
                order.append('sent')
                await asyncio.sleep(0.02)
                entry['done'] = True
                wa._network_touch()
                order.append('answered')
            asyncio.ensure_future(late_fetch())
            status = await wa._async_wait_for_network_idle()
            order.append('idle')
            return status

        assert run(scenario()) == "idle"
        assert order == ['sent', 'answered', 'idle']

    def test_a_request_that_never_finishes_hits_the_soft_cap(self):
        wa = make_wa()
        pending(wa, 'poll', url='https://shop.example/api/poll')
        started = time.time()
        assert run(wa._async_wait_for_network_idle()) == "soft-timeout:network"
        assert time.time() - started < 0.5

    def test_event_source_streams_are_not_pending(self):
        wa = make_wa()
        pending(wa, 'sse', kind='EventSource', age=1.0)
        assert run(wa._async_wait_for_network_idle()) == "idle"

    def test_a_stale_unanswered_request_is_not_pending(self):
        wa = make_wa()
        pending(wa, 'lost', age=wa._PAGE_NET_STALE_S + 1)
        assert run(wa._async_wait_for_network_idle()) == "idle"

    def test_only_fresh_activity_without_pending_requests_waits_out_the_idle_period(self):
        wa = make_wa()
        wa._PAGE_NET_IDLE_MS = 30
        wa._network_touch()
        started = time.time()
        assert run(wa._async_wait_for_network_idle()) == "idle"
        assert time.time() - started >= 0.025


@pytest.mark.unit
class TestGateRunsNetworkStage:

    def _ready_eval(self):
        async def _eval(tab, expr, **kwargs):
            return 'complete' if 'readyState' in expr else 700
        return _eval

    def test_network_stage_follows_dom_quiet(self):
        wa = make_wa()
        with patch.object(WebAccess, '_evaluate', new=AsyncMock(side_effect=self._ready_eval())), \
                patch.object(WebAccess, '_async_wait_for_network_idle',
                             new=AsyncMock(return_value='idle')) as net:
            assert run(wa._async_wait_for_page_ready(wa._tab)) == "ready"
        net.assert_awaited_once()

    def test_network_stage_runs_even_when_the_dom_never_goes_quiet(self):
        wa = make_wa()

        async def _eval(tab, expr, **kwargs):
            return 'complete' if 'readyState' in expr else 0

        with patch.object(WebAccess, '_evaluate', new=AsyncMock(side_effect=_eval)), \
                patch.object(WebAccess, '_async_wait_for_network_idle',
                             new=AsyncMock(return_value='idle')) as net:
            status = run(wa._async_wait_for_page_ready(wa._tab, quiet_timeout=0.01))
        assert status == "soft-timeout:quiet"
        net.assert_awaited_once()

    def test_network_stage_is_skipped_when_the_page_cannot_be_evaluated(self):
        wa = make_wa()
        with patch.object(WebAccess, '_evaluate', new=AsyncMock(side_effect=RuntimeError('gone'))), \
                patch.object(WebAccess, '_async_wait_for_network_idle', new=AsyncMock()) as net:
            run(wa._async_wait_for_page_ready(wa._tab))
        net.assert_not_awaited()


@pytest.mark.unit
class TestVisibleOnlyExtraction:

    def test_a_shell_with_embedded_json_has_no_visible_text(self):
        url = 'https://store.example/order'
        assert WebAccess.extract_text_with_links(SHELL, url, visible_only=True).strip() == ''
        assert 'order status' in WebAccess.extract_text_with_links(SHELL, url)

    def test_a_rendered_page_keeps_its_text(self):
        text = WebAccess.extract_text_with_links(RENDERED, 'https://store.example/order', visible_only=True)
        assert 'arrives Friday' in text


@pytest.mark.unit
class TestRegrabIfBlank:

    def test_a_blank_shell_is_read_again_after_the_network_settles(self):
        wa = make_wa()
        tab = Mock()
        tab.get_content = AsyncMock(return_value=RENDERED)
        order = []

        async def idle(self):
            order.append('network')
            return 'idle'

        with patch.object(WebAccess, '_async_wait_for_network_idle', new=idle):
            result = run(wa._async_regrab_if_blank(tab, 'https://store.example/order', SHELL))
        assert result == RENDERED
        assert order == ['network']
        tab.get_content.assert_awaited_once()

    def test_a_page_with_text_is_not_read_again(self):
        wa = make_wa()
        tab = Mock()
        tab.get_content = AsyncMock()
        with patch.object(WebAccess, '_async_wait_for_network_idle', new=AsyncMock()) as net:
            result = run(wa._async_regrab_if_blank(tab, 'https://store.example/order', RENDERED))
        assert result == RENDERED
        tab.get_content.assert_not_awaited()
        net.assert_not_awaited()

    def test_a_still_blank_re_read_is_returned_for_the_json_fallback(self):
        wa = make_wa()
        tab = Mock()
        tab.get_content = AsyncMock(return_value=SHELL)
        with patch.object(WebAccess, '_async_wait_for_network_idle', new=AsyncMock()):
            result = run(wa._async_regrab_if_blank(tab, 'https://store.example/order', SHELL))
        assert result == SHELL

    def test_fetch_via_chrome_returns_the_second_read_of_a_late_rendering_page(self):
        wa = make_wa()
        wa._tab.get = AsyncMock()
        wa._tab.get_content = AsyncMock(side_effect=[SHELL, RENDERED])
        with patch.object(WebAccess, '_async_ensure_browser', new=AsyncMock(return_value=True)), \
                patch.object(WebAccess, '_async_wait_for_page_ready', new=AsyncMock()), \
                patch.object(WebAccess, '_async_settle_cookie_banner', new=AsyncMock()), \
                patch.object(WebAccess, '_async_scroll_to_load_content', new=AsyncMock()), \
                patch.object(WebAccess, '_async_wait_for_network_idle', new=AsyncMock()):
            html = run(wa._async_fetch_via_chrome('https://store.example/order'))
        assert html == RENDERED
        assert wa._tab.get_content.await_count == 2
