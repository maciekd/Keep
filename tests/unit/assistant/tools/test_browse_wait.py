"""browse_wait: the "<pattern>, <seconds>" spec, the poll loop (URL/title/text probe,
popup following, transient probe failures, timeout) and its watchdog budget. Replaces the
sleeps the assistant hand-rolled inside browse_eval while a BankID handoff ran."""
import asyncio
import time as _time
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, patch

from assistant.tools.WebAccess import WebAccess


def make_wa(**overrides):
    wa = WebAccess.__new__(WebAccess)
    wa._tab = SimpleNamespace(name='base')
    wa._browse_session_url = 'https://shop.example/checkout'
    wa._browse_pdf_text = None
    wa._dialog_notes = []
    for k, v in overrides.items():
        setattr(wa, k, v)
    return wa


def page(url='https://shop.example/checkout', title='Checkout', text='Waiting for BankID…'):
    return {'url': url, 'title': title, 'text': text}


class FakeClock:
    """time.time() that advances a fixed step per call, so a timeout expires without sleeping."""

    strftime, localtime = staticmethod(_time.strftime), staticmethod(_time.localtime)

    def __init__(self, step):
        self.now = 1_000.0
        self.step = step

    def time(self):
        self.now += self.step
        return self.now


def wait(wa, pattern, timeout_s=30.0, *, probes, clock=None, sync=None, probed=None):
    """Run _async_browse_wait with the page probe answering `probes` in turn (an exception
    instance is raised instead of returned) and every other stage stubbed out. `probed`
    collects the tab each probe ran on."""
    def probe(tab, _js, **kw):
        if probed is not None:
            probed.append(tab)
        result = probes.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    patches = [
        patch.object(WebAccess, '_WAIT_POLL_S', 0.001),
        patch.object(WebAccess, '_async_browse_ready_midsession', new=AsyncMock(return_value=None)),
        patch.object(WebAccess, '_async_sync_popup_targets', new=sync or AsyncMock()),
        patch.object(WebAccess, '_evaluate', new=AsyncMock(side_effect=probe)),
        patch.object(WebAccess, '_async_wait_for_page_ready', new=AsyncMock()),
        patch.object(WebAccess, '_async_wait_for_settle', new=AsyncMock()),
        patch.object(WebAccess, '_async_take_snapshot', new=AsyncMock(return_value='[0] snapshot line')),
    ]
    if clock is not None:
        patches.append(patch('assistant.tools.WebAccess.time', clock))
    for p in patches:
        p.start()
    try:
        return asyncio.run(wa._async_browse_wait(pattern, timeout_s))
    finally:
        for p in patches:
            p.stop()


@pytest.mark.unit
class TestWaitSpec:

    @pytest.mark.parametrize("spec, pattern, seconds", [
        ("Continue|callback, 45", "Continue|callback", 45.0),
        ("Tack för din order", "Tack för din order", WebAccess._WAIT_DEFAULT_S),
        ("Tack, välkommen", "Tack, välkommen", WebAccess._WAIT_DEFAULT_S),   # comma in the pattern, no timeout
        ("pending, complete, 20", "pending, complete", 20.0),
        ('"done", 2.5', "done", 2.5),
        ("", "", WebAccess._WAIT_DEFAULT_S),
    ])
    def test_pattern_keeps_its_commas_only_a_trailing_number_is_the_timeout(self, spec, pattern, seconds):
        assert WebAccess._parse_wait_spec(spec) == (pattern, seconds)

    def test_multi_param_rejoin_roundtrips_a_pattern_with_a_comma(self):
        # The command parser splits browse_wait("a, b", 12) at every comma and
        # Commands re-joins the pieces; the parse must give back the original pattern.
        rejoined = ", ".join(a.strip() for a in ["a", " b", " 12"])
        assert WebAccess._parse_wait_spec(rejoined) == ("a, b", 12.0)


@pytest.mark.unit
class TestWatchdogBudget:

    def test_watchdog_outlasts_the_longest_allowed_wait(self):
        wa = make_wa()
        with patch.object(WebAccess, '_run_browse_op', return_value='ok') as op:
            wa.browse_wait("callback, 500")
        coro = op.call_args.args[1]
        coro.close()
        watchdog = op.call_args.kwargs['timeout']
        # A wait far past the cap is clamped, not honoured, and the watchdog still
        # leaves room for the closing snapshot without growing past the default budget.
        assert WebAccess._WAIT_MAX_S < watchdog <= WebAccess._WAIT_MAX_S + WebAccess._BROWSE_OP_TIMEOUT


@pytest.mark.unit
class TestWaitLoop:

    def test_match_in_the_url_returns_a_snapshot_marked_matched(self):
        wa = make_wa()
        result = wait(wa, "callback|thanks", probes=[
            page(), page(url='https://shop.example/thanks?order=KEP47L', text='Tack Marcus!')])
        assert result.startswith('MATCHED')
        assert 'thanks' in result.splitlines()[0]
        assert '[0] snapshot line' in result

    def test_match_in_the_visible_text_is_case_insensitive(self):
        wa = make_wa()
        result = wait(wa, "ORDER CONFIRMED", probes=[page(text='Your order confirmed, Marcus.')])
        assert result.startswith('MATCHED')

    def test_match_in_the_title_counts(self):
        wa = make_wa()
        result = wait(wa, "^Receipt", probes=[page(title='Receipt 4711')])
        assert result.startswith('MATCHED')

    def test_invalid_regex_is_matched_literally(self):
        wa = make_wa()
        result = wait(wa, "status (pending", probes=[page(text='status (pending) — please wait')])
        assert result.startswith('MATCHED')

    def test_a_failed_probe_is_retried_not_fatal(self):
        wa = make_wa()
        result = wait(wa, "done", probes=[asyncio.TimeoutError(), page(text='done')])
        assert result.startswith('MATCHED')

    def test_each_poll_follows_popups_and_probes_the_active_tab(self):
        wa = make_wa()
        popup = SimpleNamespace(name='popup')
        base = wa._tab

        async def open_popup_on_second_poll(_self):
            _self._tab = popup if open_popup_on_second_poll.calls else base
            open_popup_on_second_poll.calls += 1
        open_popup_on_second_poll.calls = 0

        probed = []
        result = wait(wa, "bankid/start", sync=open_popup_on_second_poll, probed=probed, probes=[
            page(), page(url='https://pay.example/bankid/start')])
        assert result.startswith('MATCHED')
        assert probed == [base, popup]

    def test_timeout_returns_the_page_and_points_at_the_network_log(self):
        wa = make_wa()
        clock = FakeClock(step=0.6)  # 1 s wait: expires on the second poll
        result = wait(wa, "bekraftelse", timeout_s=1.0, clock=clock,
                      probes=[page(), page(), page(), page()])
        assert result.startswith('TIMEOUT')
        assert 'bekraftelse' in result
        assert 'browse_network' in result
        assert '[0] snapshot line' in result

    def test_dialog_notes_are_prefixed_to_the_result(self):
        wa = make_wa(_dialog_notes=['NOTE: the page showed a confirm dialog'])
        result = wait(wa, "done", probes=[page(text='done')])
        assert result.startswith('NOTE: the page showed a confirm dialog')
        assert 'MATCHED' in result
        assert wa._dialog_notes == []

    def test_no_session_no_pattern_and_pdf_sessions_are_refused_without_polling(self):
        probe = AsyncMock()
        with patch.object(WebAccess, '_evaluate', new=probe):
            assert asyncio.run(make_wa(_browse_session_url=None)._async_browse_wait("x", 5)).startswith('ERROR')
            assert asyncio.run(make_wa()._async_browse_wait("", 5)).startswith('ERROR')
            assert asyncio.run(make_wa(_browse_pdf_text='pdf text')._async_browse_wait("x", 5)).startswith('ERROR')
        assert probe.await_count == 0
