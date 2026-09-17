"""Page readiness, cookie consent orchestration, and snapshot quality (plan 04):
soft navigation caps, DOM-quiet readiness, accept-first consent with one retry,
grouped snapshot rendering, search-field hint, and redirect warning."""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from unittest.mock import Mock, patch, AsyncMock

from assistant.tools.WebAccess import WebAccess


def run(coro):
    return asyncio.run(coro)


def make_wa(**overrides):
    wa = WebAccess.__new__(WebAccess)
    wa._PAGE_READY_POLL = 0.001
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
    for key, value in overrides.items():
        setattr(wa, key, value)
    return wa


def scripted_evaluate(ready_states, dommon_values):
    """Fake WebAccess._evaluate: pops readyState / __domMon answers per call,
    repeating the last value once a sequence is exhausted."""
    r = list(ready_states)
    d = list(dommon_values)

    async def _eval(tab, expr, **kwargs):
        seq = r if 'readyState' in expr else d
        return seq.pop(0) if len(seq) > 1 else seq[0]

    return _eval


@pytest.mark.unit
class TestWaitForPageReady:

    def test_interactive_then_quiet_is_ready(self):
        wa = make_wa()
        with patch.object(WebAccess, '_evaluate',
                          new=AsyncMock(side_effect=scripted_evaluate(['loading', 'interactive'], [700]))):
            status = run(wa._async_wait_for_page_ready(wa._tab))
        assert status == "ready"

    def test_stuck_readystate_soft_times_out_without_raising(self):
        wa = make_wa()
        with patch.object(WebAccess, '_evaluate',
                          new=AsyncMock(side_effect=scripted_evaluate(['loading'], [700]))):
            status = run(wa._async_wait_for_page_ready(wa._tab, ready_timeout=0.02))
        assert status == "soft-timeout:readyState"

    def test_dom_never_quiet_soft_times_out_without_raising(self):
        wa = make_wa()
        with patch.object(WebAccess, '_evaluate',
                          new=AsyncMock(side_effect=scripted_evaluate(['complete'], [0]))):
            status = run(wa._async_wait_for_page_ready(wa._tab, quiet_timeout=0.02))
        assert status == "soft-timeout:quiet"

    def test_missing_dommon_falls_back_to_content_stability(self):
        wa = make_wa()
        fallback = AsyncMock()
        with patch.object(WebAccess, '_evaluate',
                          new=AsyncMock(side_effect=scripted_evaluate(['complete'], [-1]))), \
                patch.object(WebAccess, '_async_wait_for_dynamic_content', new=fallback):
            status = run(wa._async_wait_for_page_ready(wa._tab))
        assert status == "fallback:no-dommon"
        fallback.assert_awaited_once()

    def test_evaluate_failure_is_soft(self):
        wa = make_wa()
        with patch.object(WebAccess, '_evaluate',
                          new=AsyncMock(side_effect=RuntimeError("renderer gone"))):
            status = run(wa._async_wait_for_page_ready(wa._tab))
        assert status == "soft-timeout:evaluate"

    def test_hanging_evaluate_is_capped_and_soft(self):
        wa = make_wa(_PAGE_READY_EVAL_TIMEOUT=0.05)

        async def hang(tab, expr, **kwargs):
            await asyncio.sleep(5)

        start = time.time()
        with patch.object(WebAccess, '_evaluate', new=AsyncMock(side_effect=hang)):
            status = run(wa._async_wait_for_page_ready(wa._tab))
        assert status == "soft-timeout:evaluate"
        assert time.time() - start < 2

    def test_quiet_wait_still_runs_after_readystate_soft_timeout(self):
        wa = make_wa()
        dommon_calls = []

        async def _eval(tab, expr, **kwargs):
            if 'readyState' in expr:
                return 'loading'
            dommon_calls.append(expr)
            return 700

        with patch.object(WebAccess, '_evaluate', new=AsyncMock(side_effect=_eval)):
            status = run(wa._async_wait_for_page_ready(wa._tab, ready_timeout=0.02))
        assert status == "soft-timeout:readyState"
        assert dommon_calls, "the quiet phase must run even after a readyState soft timeout"


@pytest.mark.unit
class TestBrowseOpenSoftNavigation:

    def _open_patches(self):
        return (
            patch.object(WebAccess, '_async_browse_ready', new=AsyncMock(return_value=None)),
            patch.object(WebAccess, '_async_close_popup_targets', new=AsyncMock()),
            patch.object(WebAccess, '_async_wait_for_page_ready', new=AsyncMock()),
            patch.object(WebAccess, '_async_settle_cookie_banner', new=AsyncMock()),
            patch.object(WebAccess, '_async_take_snapshot', new=AsyncMock(return_value="SNAP")),
        )

    def test_hanging_navigate_still_snapshots_and_never_reinits(self):
        wa = make_wa(_NAV_COMMIT_TIMEOUT=0.05)

        async def hang(*args, **kwargs):
            await asyncio.sleep(30)

        tab = Mock()
        tab.send = hang
        wa._tab = tab
        init_mock = AsyncMock()
        stop_mock = AsyncMock()
        p1, p2, p3, p4, p5 = self._open_patches()
        with p1, p2, p3, p4, p5, \
                patch.object(WebAccess, '_async_initialize_driver', new=init_mock), \
                patch.object(WebAccess, '_async_safe_stop_page_load', new=stop_mock):
            result = run(wa._async_browse_open("https://slow.example"))

        assert result == "SNAP"
        init_mock.assert_not_awaited()
        stop_mock.assert_not_awaited()
        assert wa._browse_session_url == "https://slow.example"
        assert wa._browse_fail_streak == 0

    def test_navigate_error_text_is_soft(self):
        wa = make_wa()
        wa._tab.send = AsyncMock(return_value=(Mock(), None, 'net::ERR_NAME_NOT_RESOLVED'))
        p1, p2, p3, p4, p5 = self._open_patches()
        with p1, p2, p3, p4, p5:
            result = run(wa._async_browse_open("https://nosuch.example"))
        assert result == "SNAP"

    def test_readiness_runs_before_cookie_pass(self):
        wa = make_wa()
        wa._tab.send = AsyncMock()
        order = []

        async def ready(*a, **k):
            order.append('ready')

        async def cookies(*a, **k):
            order.append('cookies')

        with patch.object(WebAccess, '_async_browse_ready', new=AsyncMock(return_value=None)), \
                patch.object(WebAccess, '_async_close_popup_targets', new=AsyncMock()), \
                patch.object(WebAccess, '_async_wait_for_page_ready', new=ready), \
                patch.object(WebAccess, '_async_settle_cookie_banner', new=cookies), \
                patch.object(WebAccess, '_async_take_snapshot', new=AsyncMock(return_value="SNAP")):
            run(wa._async_browse_open("https://shop.example"))
        assert order == ['ready', 'cookies']


@pytest.mark.unit
class TestCookieOrchestration:

    def test_retries_once_when_first_pass_matches_nothing(self):
        wa = make_wa(_COOKIE_RETRY_DELAY=0)
        remove = AsyncMock(side_effect=[{'clicked': 0, 'removed': 0},
                                        {'clicked': 1, 'removed': 2}])
        with patch.object(WebAccess, '_async_remove_cookies_banner', new=remove):
            run(wa._async_settle_cookie_banner(wa._tab, "https://shop.example"))
        assert remove.await_count == 2

    def test_no_retry_when_first_pass_clicked(self):
        wa = make_wa(_COOKIE_RETRY_DELAY=0)
        remove = AsyncMock(return_value={'clicked': 1, 'removed': 0})
        with patch.object(WebAccess, '_async_remove_cookies_banner', new=remove):
            run(wa._async_settle_cookie_banner(wa._tab, "https://shop.example"))
        assert remove.await_count == 1

    def test_no_retry_when_first_pass_removed(self):
        wa = make_wa(_COOKIE_RETRY_DELAY=0)
        remove = AsyncMock(return_value={'clicked': 0, 'removed': 3})
        with patch.object(WebAccess, '_async_remove_cookies_banner', new=remove):
            run(wa._async_settle_cookie_banner(wa._tab, "https://shop.example"))
        assert remove.await_count == 1

    def test_both_passes_empty_is_soft(self):
        wa = make_wa(_COOKIE_RETRY_DELAY=0)
        remove = AsyncMock(return_value={})
        with patch.object(WebAccess, '_async_remove_cookies_banner', new=remove):
            run(wa._async_settle_cookie_banner(wa._tab, "https://shop.example"))
        assert remove.await_count == 2

    def test_remove_returns_js_result_dict(self):
        wa = make_wa()
        with patch.object(WebAccess, '_evaluate',
                          new=AsyncMock(return_value={'clicked': 2, 'removed': 1})):
            result = run(wa._async_remove_cookies_banner(wa._tab, "https://x.example"))
        assert result == {'clicked': 2, 'removed': 1}

    def test_remove_timeout_returns_empty_dict(self):
        wa = make_wa()
        with patch.object(WebAccess, '_evaluate',
                          new=AsyncMock(side_effect=asyncio.TimeoutError())):
            assert run(wa._async_remove_cookies_banner(wa._tab, "https://x.example")) == {}

    def test_remove_failure_returns_empty_dict(self):
        wa = make_wa()
        with patch.object(WebAccess, '_evaluate',
                          new=AsyncMock(side_effect=RuntimeError("boom"))):
            assert run(wa._async_remove_cookies_banner(wa._tab, "https://x.example")) == {}

    def test_remove_non_dict_result_returns_empty_dict(self):
        wa = make_wa()
        with patch.object(WebAccess, '_evaluate', new=AsyncMock(return_value=None)):
            assert run(wa._async_remove_cookies_banner(wa._tab, "https://x.example")) == {}


@pytest.mark.unit
class TestCookieJsGuards:
    """Cheap regression tripwires for the consent JS payload."""

    def test_platform_selectors_present(self):
        js = WebAccess._COOKIE_REMOVAL_JS
        assert '#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll' in js
        assert '#CybotCookiebotDialogBodyButtonAccept' in js
        assert '#truste-consent-button' in js
        assert '#didomi-notice-agree-button' in js
        assert 'qc-cmp2-summary-buttons' in js
        assert '.cmplz-accept' in js
        assert '.cky-btn-accept' in js
        assert '#onetrust-accept-btn-handler' in js

    def test_usercentrics_shadow_dom_traversal(self):
        js = WebAccess._COOKIE_REMOVAL_JS
        assert '#usercentrics-root' in js
        assert 'shadowRoot' in js
        assert 'uc-accept-all-button' in js

    def test_swedish_and_nordic_labels_present(self):
        js = WebAccess._COOKIE_REMOVAL_JS
        for label in ('godkänn alla', 'godkänn', 'acceptera alla', 'acceptera',
                      'tillåt alla', 'tillåt', 'jag godkänner', 'jag förstår',
                      'godta alle', 'tillad alle', 'accepter alle'):
            assert label in js
        # \b breaks on å/ä/ö — the Nordic matcher must use Unicode-aware boundaries
        assert 'p{L}' in js

    def test_js_returns_click_and_removal_counts(self):
        js = WebAccess._COOKIE_REMOVAL_JS
        assert 'return { clicked: clicked, removed: removed, kept: kept }' in js

    def test_dom_monitor_js_installs_mutation_observer(self):
        js = WebAccess._DOM_MONITOR_JS
        assert '__domMon' in js
        assert 'MutationObserver' in js

    def test_generic_matcher_drops_allow_and_continue(self):
        """"allow"/"continue" match login-flow buttons — machine-gunning them
        rate-limited the x.com login. They must be gone from the generic matcher."""
        js = WebAccess._COOKIE_REMOVAL_JS
        assert '|got it|allow|continue)' not in js
        assert r"/\b(accept all|accept cookies|i agree|agree|consent|got it)\b/i" in js

    def test_generic_matcher_requires_consent_context(self):
        js = WebAccess._COOKIE_REMOVAL_JS
        assert 'const hasConsentCtx' in js
        assert 'CONSENT_CTX_RE' in js
        assert 'looksAccept && hasConsentCtx(el)' in js


@pytest.mark.unit
class TestAuthUrlConsentSkip:
    """The consent pass must never run on login/SSO flows — a stray click there
    submits the auth flow (the x.com phantom-click rate-limit incident)."""

    @pytest.mark.parametrize('url', [
        'https://x.com/i/flow/login',
        'https://x.com/i/jf/onboarding/web?mode=login',
        'https://accounts.google.com/o/oauth2/v2/auth?foo=1',
        'https://appleid.apple.com/auth/authorize',
        'https://example.com/login',
        'https://example.com/sign-in',
        'https://example.com/signup',
    ])
    def test_auth_urls_detected(self, url):
        assert WebAccess._is_auth_url(url)

    @pytest.mark.parametrize('url', [
        'https://example.com/products',
        'https://news.site/article/continue-reading-here',
        'https://shop.se/allow-list-guide',
        '',
    ])
    def test_non_auth_urls_pass_through(self, url):
        assert not WebAccess._is_auth_url(url)

    def test_settle_skips_remove_on_auth_url(self):
        wa = make_wa(_COOKIE_RETRY_DELAY=0)
        remove = AsyncMock(return_value={'clicked': 0, 'removed': 0})
        with patch.object(WebAccess, '_async_remove_cookies_banner', new=remove):
            run(wa._async_settle_cookie_banner(wa._tab, 'https://x.com/i/flow/login'))
        remove.assert_not_awaited()

    def test_settle_still_runs_on_normal_url(self):
        wa = make_wa(_COOKIE_RETRY_DELAY=0)
        remove = AsyncMock(return_value={'clicked': 1, 'removed': 0})
        with patch.object(WebAccess, '_async_remove_cookies_banner', new=remove):
            run(wa._async_settle_cookie_banner(wa._tab, 'https://shop.example'))
        remove.assert_awaited()


@pytest.mark.unit
class TestUserAgentFingerprint:
    """The UA override must scrub the HeadlessChrome bot tell and keep the
    Sec-Ch-Ua brand versions consistent with the real Chrome version."""

    def _tab_reporting(self, ua):
        tab = Mock()
        tab.send = AsyncMock(return_value=('1.3', 'HeadlessChrome/151.0.7922.174',
                                           'rev', ua, 'v8'))
        return tab

    def test_resolve_ua_strips_headless_and_parses_version(self):
        wa = make_wa(_UA_FALLBACK_MAJOR='151')
        wa._ua_cache = None
        ua_in = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                 'AppleWebKit/537.36 (KHTML, like Gecko) '
                 'HeadlessChrome/151.0.7922.174 Safari/537.36')
        ua, major, full = run(wa._async_resolve_ua(self._tab_reporting(ua_in)))
        assert 'HeadlessChrome' not in ua
        assert 'Chrome/151.0.7922.174' in ua
        assert (major, full) == ('151', '151.0.7922.174')

    def test_resolve_ua_caches(self):
        wa = make_wa(_UA_FALLBACK_MAJOR='151')
        wa._ua_cache = None
        tab = self._tab_reporting('Mozilla/5.0 Chrome/151.0.7922.174 Safari/537.36')
        run(wa._async_resolve_ua(tab))
        tab.send.reset_mock()
        run(wa._async_resolve_ua(tab))
        assert tab.send.await_count == 0

    def test_resolve_ua_falls_back_when_getversion_fails(self):
        wa = make_wa(_UA_FALLBACK_MAJOR='151')
        wa._ua_cache = None
        tab = Mock()
        tab.send = AsyncMock(side_effect=RuntimeError('no transport'))
        ua, major, full = run(wa._async_resolve_ua(tab))
        assert 'HeadlessChrome' not in ua
        assert major == '151'

    def test_apply_tweaks_sends_ua_override_and_headers(self):
        wa = make_wa(_UA_FALLBACK_MAJOR='151')
        wa._ua_cache = None
        tab = self._tab_reporting(
            'Mozilla/5.0 (Macintosh) HeadlessChrome/151.0.7922.174 Safari/537.36')
        run(wa._async_apply_anti_bot_tweaks(tab))
        # UA override + extra headers + stealth script were all dispatched.
        assert tab.send.await_count >= 3

    def _wa(self):
        return WebAccess.__new__(WebAccess)

    def test_all_main_renders_without_separator(self):
        records = [{'tag': 'button', 'label': 'A', 'region': 'main'},
                   {'tag': 'button', 'label': 'B'}]
        lines = self._wa()._render_snapshot_lines(records)
        assert lines == ['[0] button "A"', '[1] button "B"']

    def test_large_nav_group_collapses_to_one_line_with_all_indices(self):
        records = [{'tag': 'button', 'label': 'Buy', 'region': 'main'}]
        records += [{'tag': 'a', 'label': f'Menu {i}', 'href': '/x', 'region': 'nav'}
                    for i in range(6)]
        lines = self._wa()._render_snapshot_lines(records)
        assert lines[0] == '[0] button "Buy"'
        assert lines[1] == ''
        assert lines[2] == '--- header/nav/footer (collapsed) ---'
        nav_line = lines[3]
        assert nav_line.startswith('nav: 6 items — ')
        for i in range(1, 7):
            assert f'[{i}] Menu {i - 1}' in nav_line
        assert len(lines) == 4

    def test_small_group_renders_full_lines_after_separator(self):
        records = [{'tag': 'button', 'label': 'Buy', 'region': 'main'},
                   {'tag': 'a', 'label': 'Home', 'href': '/', 'region': 'nav'},
                   {'tag': 'a', 'label': 'About', 'href': '/about', 'region': 'nav'}]
        lines = self._wa()._render_snapshot_lines(records)
        assert lines == ['[0] button "Buy"', '',
                         '--- header/nav/footer (collapsed) ---',
                         '[1] link "Home" -> /',
                         '[2] link "About" -> /about']

    def test_main_records_listed_first_indices_preserved(self):
        records = [{'tag': 'a', 'label': 'Logo', 'href': '/', 'region': 'header'},
                   {'tag': 'input', 'type': 'text', 'label': 'Name', 'region': 'main'},
                   {'tag': 'a', 'label': 'Legal', 'href': '/legal', 'region': 'footer'},
                   {'tag': 'button', 'label': 'Send', 'region': 'main'}]
        lines = self._wa()._render_snapshot_lines(records)
        assert lines[0] == '[1] input[text] "Name"'
        assert lines[1] == '[3] button "Send"'
        assert '--- header/nav/footer (collapsed) ---' in lines
        assert '[0] link "Logo" -> /' in lines
        assert '[2] link "Legal" -> /legal' in lines

    def test_aside_group_collapses_too(self):
        records = [{'tag': 'a', 'label': f'Related {i}', 'href': '/r', 'region': 'aside'}
                   for i in range(6)]
        lines = self._wa()._render_snapshot_lines(records)
        assert lines[0] == '--- header/nav/footer (collapsed) ---'
        assert lines[1].startswith('aside: 6 items — ')

    def test_collapsed_non_link_items_show_their_kind(self):
        # A collapsed header once rendered a search *button* indistinguishably
        # from an input ("[2] Sök efter produkter"), drawing doomed fill attempts.
        records = [{'tag': 'a', 'label': 'Home', 'href': '/', 'region': 'header'},
                   {'tag': 'button', 'label': 'Sök', 'region': 'header'},
                   {'tag': 'input', 'type': 'search', 'label': 'Sök fält', 'region': 'header'},
                   {'tag': 'select', 'label': 'Lang', 'region': 'header'},
                   {'tag': 'div', 'role': 'textbox', 'label': 'Note', 'region': 'header'},
                   {'tag': 'a', 'label': 'Help', 'href': '/help', 'region': 'header'}]
        lines = self._wa()._render_snapshot_lines(records)
        header_line = lines[-1]
        assert '[0] Home' in header_line          # links stay bare
        assert '[1] button Sök' in header_line
        assert '[2] input[search] Sök fält' in header_line
        assert '[3] select Lang' in header_line
        assert '[4] editable Note' in header_line
        assert '[5] Help' in header_line

    def test_collapsed_labels_are_truncated(self):
        long_label = 'X' * 60
        records = [{'tag': 'a', 'label': long_label, 'href': '/x', 'region': 'nav'}
                   for _ in range(6)]
        lines = self._wa()._render_snapshot_lines(records)
        assert long_label not in lines[-1]
        assert 'X' * 39 + '…' in lines[-1]

    def test_iframe_source_suffix_kept_in_main_lines(self):
        records = [{'tag': 'button', 'label': 'Pay', 'source': 'pay.klarna.com/frame'}]
        lines = self._wa()._render_snapshot_lines(records)
        assert lines == ['[0] button "Pay"  [in pay.klarna.com/frame]']

    def test_group_order_is_header_nav_aside_footer(self):
        records = [{'tag': 'a', 'label': f'F{i}', 'href': '/', 'region': 'footer'} for i in range(6)]
        records += [{'tag': 'a', 'label': f'H{i}', 'href': '/', 'region': 'header'} for i in range(6)]
        records += [{'tag': 'a', 'label': f'N{i}', 'href': '/', 'region': 'nav'} for i in range(6)]
        lines = self._wa()._render_snapshot_lines(records)
        group_lines = [l.split(':')[0] for l in lines if ': 6 items' in l]
        assert group_lines == ['header', 'nav', 'footer']


@pytest.mark.unit
class TestPageContextHeader:

    def test_h1_and_headings(self):
        ctx = {'h1': 'Veckans erbjudanden', 'headings': ['Bästsäljare just nu', 'Populära varumärken']}
        assert WebAccess._format_page_context(ctx) == \
            'Page: h1 "Veckans erbjudanden" | headings: "Bästsäljare just nu", "Populära varumärken"\n'

    def test_h1_only(self):
        assert WebAccess._format_page_context({'h1': 'Kassa'}) == 'Page: h1 "Kassa"\n'

    def test_headings_only(self):
        assert WebAccess._format_page_context({'headings': ['A']}) == 'Page: headings: "A"\n'

    def test_empty_context_renders_nothing(self):
        assert WebAccess._format_page_context({}) == ''
        assert WebAccess._format_page_context(None) == ''
        assert WebAccess._format_page_context('junk') == ''


@pytest.mark.unit
class TestSearchFieldHint:

    def _wa(self):
        return WebAccess.__new__(WebAccess)

    HINT = 'search field: filling does not submit'

    def test_type_search_gets_hint(self):
        line = self._wa()._format_snapshot_line(2, {'tag': 'input', 'type': 'search', 'label': 'Sök'})
        assert self.HINT in line
        assert 'browse_key("Enter")' in line
        assert '\\n' in line

    def test_is_search_flag_gets_hint(self):
        line = self._wa()._format_snapshot_line(
            0, {'tag': 'input', 'type': 'text', 'label': 'Sök produkt', 'isSearch': True})
        assert self.HINT in line

    def test_plain_text_input_has_no_hint(self):
        line = self._wa()._format_snapshot_line(0, {'tag': 'input', 'type': 'text', 'label': 'Namn'})
        assert self.HINT not in line

    def test_checkbox_never_gets_hint(self):
        line = self._wa()._format_snapshot_line(
            0, {'tag': 'input', 'type': 'checkbox', 'label': 'Sök', 'checked': False, 'isSearch': True})
        assert self.HINT not in line


@pytest.mark.unit
class TestUrlMismatchWarning:

    def test_urls_differ_on_path_change(self):
        assert WebAccess._urls_differ('https://shop.example/sok', 'https://www.shop.example/')

    def test_benign_redirects_are_quiet(self):
        assert not WebAccess._urls_differ('http://shop.example', 'https://www.shop.example/')
        assert not WebAccess._urls_differ('https://shop.example/a/', 'https://shop.example/a')
        assert not WebAccess._urls_differ('https://shop.example/a?q=1', 'https://shop.example/a?utm=x')

    def _snapshot(self, wa, requested_url):
        async def fake_eval(tab, expr, **kwargs):
            if 'document.title' in expr:
                return 'Shop'
            return {'records': [{'tag': 'input', 'type': 'search', 'label': 'Sök'}],
                    'crossOriginFrames': [],
                    'context': {'h1': 'Veckans erbjudanden'}}

        with patch.object(WebAccess, '_evaluate', new=AsyncMock(side_effect=fake_eval)), \
                patch.object(WebAccess, '_async_sync_popup_targets', new=AsyncMock()), \
                patch.object(WebAccess, '_async_list_frame_targets', new=AsyncMock(return_value=[])):
            return run(wa._async_take_snapshot(requested_url=requested_url))

    def test_snapshot_warns_on_silent_redirect(self):
        wa = make_wa()
        wa._tab = SimpleNamespace(target=SimpleNamespace(url='https://www.shop.example/'))
        result = self._snapshot(wa, 'https://www.shop.example/sok')
        assert result.startswith('WARNING: the browser ended up on https://www.shop.example/')
        assert 'https://www.shop.example/sok' in result

    def test_snapshot_quiet_when_urls_match(self):
        wa = make_wa()
        wa._tab = SimpleNamespace(target=SimpleNamespace(url='https://www.shop.example/sok'))
        result = self._snapshot(wa, 'https://shop.example/sok')
        assert 'WARNING' not in result
        assert 'Page: h1 "Veckans erbjudanden"' in result
        assert 'search field: filling does not submit' in result


@pytest.mark.unit
class TestCollectorJsGuards:
    """Regression tripwires for the snapshot collector's new features."""

    def test_collector_dedups_regions_and_context(self):
        js = WebAccess._BROWSE_COLLECT_JS
        assert 'dedupNested' in js
        assert 'regionOf' in js
        assert 'pageContext' in js
        assert 'isSearch' in js
        assert "context: pageContext(document)" in js

    def test_dedup_runs_before_index_tagging(self):
        js = WebAccess._BROWSE_COLLECT_JS
        assert js.index('all = dedupNested(all)') < js.index("setAttribute('data-floxy-snap-idx'")

    def test_handler_bound_controls_are_collected(self):
        # Div/span widgets bound with addEventListener (booking calendars, qty
        # pickers) carry no role/href/onclick: the collector must pick them up
        # from cursor:pointer or a choice-naming data-* attribute, and flag them.
        js = WebAccess._BROWSE_COLLECT_JS
        assert 'function handlerBound' in js
        assert "cursor === 'pointer'" in js
        for attr in ('data-date', 'data-qty', 'data-time', 'data-value'):
            assert f"'{attr}'" in js
        assert 'rec.clickable = true' in js
        assert 'rec.data = hint' in js
        assert 'rec.cls = cls' in js
        # data-id alone is too common to count without the cursor
        assert "choice.indexOf('data-id=') === 0" in js
        # never a container that wraps a real control, nor something inside one
        assert 'el.closest(SEL) || el.querySelector(SEL)' in js
        assert js.index('handlerBound(doc, win)') < js.index('if (!isInteractive(el, win)) continue;')


@pytest.mark.unit
class TestClickableSnapshotLines:
    def _wa(self):
        return WebAccess.__new__(WebAccess)

    def test_clickable_with_data_and_class(self):
        rec = {'tag': 'span', 'type': '', 'role': '', 'label': 'September 2026 / 12',
               'clickable': True, 'data': 'data-date=2026-09-12', 'cls': 'day av wait'}
        assert self._wa()._format_snapshot_line(9, rec) == \
            '[9] clickable span "September 2026 / 12" (data-date=2026-09-12, class="day av wait")'

    def test_clickable_without_hints(self):
        rec = {'tag': 'div', 'type': '', 'role': '', 'label': 'Change language', 'clickable': True}
        assert self._wa()._format_snapshot_line(2, rec) == '[2] clickable div "Change language"'

    def test_plain_div_record_unchanged(self):
        rec = {'tag': 'div', 'type': '', 'role': '', 'label': 'x'}
        assert self._wa()._format_snapshot_line(0, rec) == '[0] div "x"'

    def test_group_kind_marks_clickable(self):
        rec = {'tag': 'span', 'label': '4', 'clickable': True, 'data': 'data-qty=4'}
        assert self._wa()._collapsed_item_kind(rec) == 'clickable span'


@pytest.mark.unit
class TestMonitorInjection:
    """Both monitors must actually be injected on every new document: without
    _DOM_MONITOR_JS on the page, _async_wait_for_page_ready silently degrades to
    the no-dommon content-stability fallback forever."""

    @staticmethod
    def _sent_scripts(tab):
        scripts = []
        for call in tab.send.await_args_list:
            cdp_obj = call.args[0]
            assert cdp_obj.gi_code.co_name == 'add_script_to_evaluate_on_new_document'
            scripts.append(cdp_obj.send(None)['params']['source'])
        return scripts

    def test_dom_monitor_injected_on_new_document(self):
        wa = make_wa()
        tab = Mock()
        tab.send = AsyncMock()
        run(wa._async_setup_network_monitor(tab))
        scripts = self._sent_scripts(tab)
        assert len(scripts) == 1
        assert '__domMon' in scripts[0]

    def test_injection_failure_is_soft(self):
        wa = make_wa()
        tab = Mock()
        tab.send = AsyncMock(side_effect=Exception("Page.addScriptToEvaluateOnNewDocument failed"))
        run(wa._async_setup_network_monitor(tab))  # must not raise
        assert tab.send.await_count == 1
