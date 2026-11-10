"""The text fetch's lighter readiness gate: `open_url` runs the browse session's gate without
the network stage, with a shorter DOM-quiet cap and a single consent pass, while a browse
session keeps every wait it had."""
import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from assistant.tools.WebAccess import WebAccess
from tests.unit.assistant.tools.test_page_readiness import make_wa, run, scripted_evaluate


@pytest.mark.unit
class TestNetworkStageSkipped:
    @pytest.mark.parametrize("ready_states, dommon_values, status", [
        (['complete'], [700], "ready"),
        (['complete'], [0], "soft-timeout:quiet"),
        (['complete'], [-1], "fallback:no-dommon"),
    ])
    def test_no_exit_of_the_gate_waits_for_the_network_when_told_not_to(self, ready_states, dommon_values, status):
        wa = make_wa()
        net = AsyncMock(return_value='idle')
        with patch.object(WebAccess, '_evaluate', new=AsyncMock(side_effect=scripted_evaluate(ready_states, dommon_values))), \
                patch.object(WebAccess, '_async_wait_for_dynamic_content', new=AsyncMock()), \
                patch.object(WebAccess, '_async_wait_for_network_idle', new=net):
            assert run(wa._async_wait_for_page_ready(wa._tab, quiet_timeout=0.01, network=False)) == status
        net.assert_not_awaited()

    def test_the_stage_is_still_run_by_default(self):
        wa = make_wa()
        net = AsyncMock(return_value='idle')
        with patch.object(WebAccess, '_evaluate', new=AsyncMock(side_effect=scripted_evaluate(['complete'], [700]))), \
                patch.object(WebAccess, '_async_wait_for_network_idle', new=net):
            run(wa._async_wait_for_page_ready(wa._tab))
        net.assert_awaited_once()


@pytest.mark.unit
class TestOnePassConsent:
    def test_nothing_matched_is_not_re_checked_and_costs_no_pause(self):
        wa = make_wa(_COOKIE_RETRY_DELAY=1.0)
        remove = AsyncMock(return_value={'clicked': 0, 'removed': 0})
        with patch.object(WebAccess, '_async_remove_cookies_banner', new=remove), \
                patch('assistant.tools.WebAccess.asyncio.sleep', new=AsyncMock()) as sleep:
            run(wa._async_settle_cookie_banner(wa._tab, "https://news.example", retry=False))
        assert remove.await_count == 1
        sleep.assert_not_awaited()

    def test_a_kept_only_result_is_still_traced_without_the_retry(self):
        wa = make_wa(_COOKIE_RETRY_DELAY=0)
        remove = AsyncMock(return_value={'clicked': 0, 'removed': 0, 'kept': 3})
        with patch.object(WebAccess, '_async_remove_cookies_banner', new=remove), \
                patch('assistant.tools.WebAccess.trace') as trace:
            run(wa._async_settle_cookie_banner(wa._tab, "https://news.example", retry=False))
        assert remove.await_count == 1
        assert any('kept 3' in c.args[0] for c in trace.call_args_list), "the refused matches stay in the trace for debugging"

    def test_an_auth_url_is_skipped_whether_or_not_a_retry_is_wanted(self):
        wa = make_wa()
        remove = AsyncMock()
        with patch.object(WebAccess, '_async_remove_cookies_banner', new=remove):
            run(wa._async_settle_cookie_banner(wa._tab, "https://x.com/i/flow/login", retry=False))
        remove.assert_not_awaited()


@pytest.mark.unit
class TestWhichPathGetsWhichGate:
    def test_the_fetch_runs_the_light_gate(self):
        wa = make_wa()
        wa._tab.get = AsyncMock()
        wa._tab.get_content = AsyncMock(return_value="<html><body><p>Headlines of the day</p></body></html>")
        ready, cookies = AsyncMock(), AsyncMock()
        with patch.object(WebAccess, '_async_ensure_browser', new=AsyncMock(return_value=True)), \
                patch.object(WebAccess, '_async_wait_for_page_ready', new=ready), \
                patch.object(WebAccess, '_async_settle_cookie_banner', new=cookies), \
                patch.object(WebAccess, '_async_scroll_to_load_content', new=AsyncMock()):
            run(wa._async_fetch_via_chrome('https://news.example/'))
        assert ready.await_args.kwargs['network'] is False
        assert ready.await_args.kwargs['quiet_timeout'] < WebAccess._PAGE_QUIET_TIMEOUT, "a shorter DOM-quiet cap than the browse session's"
        assert cookies.await_args.kwargs['retry'] is False

    def test_a_browse_session_keeps_the_full_gate(self):
        wa = make_wa()
        wa._tab.send = AsyncMock()
        ready, cookies = AsyncMock(), AsyncMock()
        with patch.object(WebAccess, '_async_browse_ready', new=AsyncMock(return_value=None)), \
                patch.object(WebAccess, '_async_close_popup_targets', new=AsyncMock()), \
                patch.object(WebAccess, '_async_wait_for_page_ready', new=ready), \
                patch.object(WebAccess, '_async_settle_cookie_banner', new=cookies), \
                patch.object(WebAccess, '_async_take_snapshot', new=AsyncMock(return_value="SNAP")):
            run(wa._async_browse_open("https://shop.example"))
        assert ready.await_args.kwargs.get('network', True) is True
        assert 'quiet_timeout' not in ready.await_args.kwargs
        assert cookies.await_args.kwargs.get('retry', True) is True
