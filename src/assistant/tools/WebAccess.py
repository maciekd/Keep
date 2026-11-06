import asyncio
import atexit
import base64
import concurrent.futures
import io
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import threading
import time
from datetime import datetime
import requests
from typing import Optional, ClassVar, List
from urllib.parse import urljoin, urlparse, urlunparse, urlencode, parse_qs, parse_qsl

from . import PdfExtractor, SnapshotFormat

import cloudscraper
from curl_cffi import requests as cffi_requests
import psutil
import nodriver
import websockets.exceptions
from nodriver import cdp
from nodriver.core.connection import ProtocolException, Transaction
from bs4 import BeautifulSoup, Comment, NavigableString

from ..integration.Integration import Integration
from common.logging import info, trace, note, fine, error
from common.ConfigManager import ConfigManager
from common.Paths import CACHE_DIR, LOG_DIR


# Exceptions that mean "the CDP transport / Chrome itself is dead", as opposed to a
# page-level failure. WebSocketException covers InvalidStatus/InvalidHandshake/
# ConnectionClosed — it subclasses Exception directly, so without it here a rejected
# websocket handshake (e.g. "HTTP 500") would fall through to generic handlers and
# never trigger a reinit.
# Opens every browse_network result; the memory curator drops the listings from its transcript by it.
NETWORK_LOG_HEADER = "Network log — "

_BROWSER_DEAD_EXC = (
    ProtocolException,
    ConnectionError,
    OSError,
    asyncio.TimeoutError,
    websockets.exceptions.WebSocketException,
)


class BrowseWatchdogTimeout(Exception):
    """A whole browse command exceeded the sync-side watchdog and was cancelled.

    Distinct from asyncio.TimeoutError (a single bounded CDP call expiring) so the
    sync wrappers can tell "one stage was slow, the command handled it" from "the
    command itself is gone and the session needs a state reset"."""


class _LoopThread:
    """Owns a private asyncio event loop running on a dedicated daemon thread.

    Sync callers submit coroutines via `submit(coro, timeout)`; the call blocks
    until the coroutine completes (or the timeout elapses) and propagates exceptions.
    All nodriver calls go through this thread so the rest of the codebase stays sync.
    """

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    def start(self):
        if self._thread is not None:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="WebAccessLoop")
        self._thread.start()
        self._ready.wait(timeout=5)

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            try:
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                self._loop.close()
            except Exception:
                pass

    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        return self._loop

    def is_running(self) -> bool:
        return self._loop is not None and self._loop.is_running()

    def submit(self, coro, timeout=None):
        if not self.is_running():
            raise RuntimeError("WebAccess loop is not running")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            # On 3.12 concurrent.futures.TimeoutError IS the builtin TimeoutError,
            # so a coroutine's own asyncio.TimeoutError lands here too. A done
            # future means the coroutine finished on its own (with that error, or
            # just under the wire) — that outcome belongs to the caller, not the
            # watchdog: converting it would trigger a spurious session reset and
            # a misleading "cancelled after Ns" report.
            if fut.done():
                return fut.result()
            # Abandoning the future would leave the coroutine running on the loop
            # forever — holding the browse lock and interleaving with the next
            # command on the same tab (the 2026-08-31 checkout wedge). Cancel it so
            # the session actually frees; inner CDP calls are shielded
            # (_cdp_wait_for), so cancellation cannot corrupt the websocket listener.
            fut.cancel()
            raise BrowseWatchdogTimeout(
                f"operation exceeded {timeout:.0f}s and was cancelled") from None

    def stop(self):
        if self.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        self._loop = None
        self._thread = None
        self._ready = threading.Event()


class WebAccess(Integration):
    """Web search and browsing: reads pages and clicks through sites itself.

    Gives Keep the web: it searches, reads pages and clicks through a site in a browser of its
    own when a task needs it. It needs no credentials and is on unless switched off.

    Drives a headless Chrome via nodriver for the browse_* commands, web search and page
    fetches. Singleton; the browser lives on a dedicated asyncio loop thread and is started
    lazily."""
    _instance: ClassVar[Optional['WebAccess']] = None

    @classmethod
    def get_instance(cls) -> 'WebAccess':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        if cls._instance is not None:
            cls._instance._shutdown_driver()
        cls._instance = None

    @property
    def name(self) -> str:
        return "Web Access"

    enabled_by_default = True   # no credentials, and the assistant leans on it constantly

    @property
    def config_keys(self) -> List[str]:
        return []

    def check_health(self) -> bool:
        if self._browser_closed_by_user:
            # browse_close deliberately shut Chrome down; a fresh browser starts
            # lazily on the next browse command, so this is not a failure state.
            return self._driver_ready.is_set()
        return self._driver_ready.is_set() and self._browser is not None

    def wait_ready(self, timeout: float) -> bool:
        """Chrome starts on the loop thread after the constructor returns; ready once that
        attempt has finished, either way."""
        return self._driver_ready.wait(timeout)

    # Gap between the two nodriver.start() attempts in _async_initialize_driver.
    # Without this, both retries hit the same back-pressure window from the dying
    # Chrome process tree and fail with "Failed to connect to browser".
    _INIT_INTER_RETRY_DELAY = 15.0
    # Minimum gap between successive recovery attempts after init has failed,
    # to avoid hammering Chrome when something is broken at the OS level.
    _INIT_RETRY_BACKOFF = 60.0

    # Circuit breaker: consecutive infra failures before one forced (backoff-
    # bypassing) restart is attempted; if that also fails the hard-down latch
    # makes every browse command fail fast until a later init succeeds.
    _BREAKER_TRIP_THRESHOLD = 3
    _HARD_DOWN_ERROR = ("ERROR: browser is unavailable on this server — "
                        "do not retry browse commands in this request.")
    _BROWSER_ALIVE_PROBE_TIMEOUT = 3.0

    # Fresh popup targets still sitting on about:blank: how long the sync polls
    # inline for them to navigate (covers window.open popups like Klarna/BankID
    # that open blank and navigate within milliseconds), and how long they may
    # stay pending across syncs before being forgotten.
    _POPUP_BLANK_GRACE = 2.0
    _POPUP_BLANK_POLL = 0.25
    _POPUP_PENDING_EXPIRY = 10.0

    # Network capture (browse_network): the XHR/fetch/document traffic of the base
    # tab and every adopted popup, kept per browse session. Bodies are read only
    # for text-like XHR/fetch responses and only right after loading finishes —
    # Chrome evicts them soon after. Without this, the 2026-09-26 Klarna BankID
    # flow gave nothing to tell "signed but rejected" from "never completed".
    _NETWORK_LOG_MAX = 80
    _NETWORK_CAPTURE_TYPES = frozenset({'XHR', 'Fetch', 'Document', 'EventSource'})
    # Analytics, ad-conversion and error-reporting beacons are never captured: they tell the
    # LLM nothing about the page, their kilometre-long query strings matched the 2026-10-22
    # Tesla checkout's browse_network("cart|order") filter and buried the one failing POST,
    # they ate the ring buffer, and a Facebook pixel that never answers held the network-idle
    # wait at its cap on every step. Payment and auth hosts (Adyen, Stripe, Klarna, BankID…)
    # are not beacons and stay. Header-bidding networks (Primis, Unruly, Publift, Amazon TAM, Lotame)
    # re-auction an idle page's ad slots every few seconds for as long as the tab is open: a
    # Songkick tab left by the 2026-11-04 watch check wrote 500 of them in twenty minutes.
    _NETWORK_NOISE_URL_RE = re.compile(
        r'^https?://(?:[^/]*\.)?(?:'
        r'facebook\.com/tr\b|connect\.facebook\.net|google-analytics\.com|analytics\.google\.com|'
        r'googletagmanager\.com|googleadservices\.com|doubleclick\.net|googlesyndication\.com|'
        r'google\.[a-z.]{2,6}/(?:pagead|ccm)/|bat\.bing\.com|clarity\.ms|hotjar\.com|'
        r'segment\.(?:io|com)|mixpanel\.com|amplitude\.com|sentry\.io|datadoghq\.com|nr-data\.net|'
        r'analytics\.tiktok\.com|ads\.linkedin\.com|tr\.snapchat\.com|ct\.pinterest\.com|'
        r'criteo\.(?:com|net)|adnxs\.com|klaviyo\.com|fullstory\.com|'
        r'primis\.tech|primis-tech\.org|unrulymedia\.com|pub\.network|amazon-adsystem\.com|crwdcntrl\.net'
        r')', re.I)
    _NETWORK_BODY_TYPES = frozenset({'XHR', 'Fetch', 'EventSource'})
    _NETWORK_BODY_MIME_RE = re.compile(r'json|text/|javascript|xml|x-www-form-urlencoded', re.I)
    _NETWORK_BODY_MAX = 1500          # chars of a body kept for browse_network
    # Request fields whose value is a credential by name: masked before the body is stored,
    # traced or shown, whatever the vault knows. Short names are bounded so "shipping" is not a pin.
    _NETWORK_SECRET_KEY_RE = re.compile(
        r'pass(?:word|wd|phrase|code)?|pwd|secret|token|credential|api[_-]?key|private|personnummer|'
        r'social[_-]?security|card[_-]?(?:number|num|no)|(?<![a-z])(?:pin|otp|cvv|cvc|ssn|tan)(?![a-z])', re.I)
    _NETWORK_SECRET_MASK = '•••'
    # A key, "=", then anything without whitespace. Keep it this flat: spelling out the `&key=value`
    # repetition gave the engine several ways to split every field, and a body that then failed to
    # match (a space in the last value) backtracked exponentially with the GIL held — the
    # 2026-10-05 outage, where one tracker POST on a news page froze the whole server.
    _NETWORK_FORM_BODY_RE = re.compile(r'^[\w.%+\-\[\]]+=\S*$')
    _NETWORK_BODY_TRACE_MAX = 400     # chars of a body written to the log
    _NETWORK_BODY_FETCH_TIMEOUT = 3.0
    _NETWORK_REPORT_DEFAULT = 40      # entries browse_network returns without a filter
    _NETWORK_REPORT_MAX_CHARS = 14000

    # browse_wait: poll the page (URL, title, visible text) for a pattern. Capped so
    # the wait plus its closing snapshot stays inside the command watchdog.
    _WAIT_DEFAULT_S = 30.0
    _WAIT_MAX_S = 60.0
    _WAIT_POLL_S = 1.0
    _WAIT_PROBE_JS = ("(function(){ return {url: location.href, title: document.title || '', "
                      "text: document.body ? (document.body.innerText || '') : ''}; })()")

    # Page-readiness caps. ALL of these are SOFT: expiry logs and proceeds with
    # whatever the page has rendered — never raises, never window.stop(), never
    # a driver reinit because a page is slow. See _async_wait_for_page_ready.
    _NAV_COMMIT_TIMEOUT = 8.0       # cap on the Page.navigate CDP response in browse_open
    _PAGE_READY_TIMEOUT = 10.0      # cap on polling document.readyState
    _PAGE_QUIET_MS = 600            # DOM must be mutation-free this long to count as quiet
    _PAGE_QUIET_TIMEOUT = 2.0       # cap on waiting for DOM quiet (a spinner never goes quiet)
    _PAGE_READY_EVAL_TIMEOUT = 2.0  # cap on each individual in-page evaluate while polling
    _PAGE_READY_POLL = 0.2          # readyState / DOM-quiet polling interval
    # A quiet DOM is not a finished page: Apple's order page (2026-10-15) is a static
    # shell that solves a challenge, then fetches the order and renders it — the grab
    # beat the fetch by 0.4 s. So after DOM quiet the gate also waits for the captured
    # XHR/fetch traffic to settle. Soft like the rest: a polling page hits the cap.
    _PAGE_NET_IDLE_MS = 500         # no request sent or answered this long counts as idle
    _PAGE_NET_IDLE_TIMEOUT = 4.0    # cap on waiting for the network to go idle
    _PAGE_NET_STALE_S = 30.0        # an unanswered request older than this is a long-poll or a lost event, not pending
    _BLANK_REGRAB_WAIT_S = 2.0      # fetch_via_chrome: pause before re-reading a page that rendered no text

    # Hard caps on individual CDP round-trips inside browse commands. A renderer
    # process swap (clicking into a Shopify checkout) or an OOPIF destroyed by
    # navigation can leave a Runtime.evaluate / flat-session command with no reply
    # forever; these caps turn such wedges into per-stage errors the command
    # handles (skip the frame, report the stage) instead of hanging the whole
    # command until the watchdog.
    _EVAL_TIMEOUT = 10.0            # any single Runtime.evaluate on the main tab
    _FRAME_ATTACH_TIMEOUT = 5.0     # Target.attachToTarget for an OOPIF session
    _FRAME_SEND_TIMEOUT = 10.0      # any single flat-session command to an OOPIF
    _SNAPSHOT_FRAME_BUDGET = 12.0   # total OOPIF collection time per snapshot
    # An OOPIF the parent DOM already shows (a Stripe card field, an hCaptcha
    # challenge) gets its own CDP target only ~0.5s after the parent went quiet;
    # the snapshot polls the target list this long for frames it can see in the
    # DOM but not yet in Target.getTargets before collecting.
    _OOPIF_SETTLE_TIMEOUT = 2.0
    _OOPIF_SETTLE_POLL = 0.25

    # Sync-side watchdog on whole browse commands. Must exceed the worst-case SUM
    # of the sanctioned stage budgets above (~80s: nav commit + readiness + DOM
    # quiet + consent evals + main eval + OOPIF budget, plus the mid-session
    # Chrome-restart path in browse_inspect) — a slow-but-healthy page working
    # through its soft caps must never trip it, or the reset/reopen loop is worse
    # than the wait. On expiry the coroutine is cancelled and the browse session
    # state reset (was 120s with no cancellation — the coroutine kept running and
    # wedged every successor; then 45s, smaller than its own stage budgets).
    _BROWSE_OP_TIMEOUT = 90.0
    _POPUP_SETTLE_S = 0.3   # after a click/Enter: a beat for Chrome to create a window.open target
    # After a click/Enter the old document sits quietly while the request it fired is in
    # flight, so the readiness gate alone passes on the page the LLM already saw. The page
    # is watched through its text signature (URL + a hash of body.innerText — a Material
    # ripple mutates the DOM on every click, a fill changes no text, a real transition
    # always does): first wait (briefly) for the signature to move or a navigation to
    # start, then for the page to settle — signature steady between two probes and no
    # visible progress indicator — and if the signature still equals the one before the
    # action, give it one more wait before re-reading and flagging it as unchanged.
    _ACTION_CHANGE_TIMEOUT = 1.5
    _ACTION_CHANGE_POLL = 0.1
    _ACTION_UNCHANGED_RETRY_S = 1.0
    _SETTLE_TIMEOUT = 3.0
    _SETTLE_POLL = 0.25
    _PAGE_PROBE_JS = """
(function() {
    var t = document.body ? (document.body.innerText || '') : '';
    var h = 0;
    for (var i = 0; i < t.length; i++) h = (h * 31 + t.charCodeAt(i)) | 0;
    var busy = false;
    try {
        var sel = '[role="progressbar"], progress, mat-spinner, mat-progress-spinner, mat-progress-bar, ' +
                  '.spinner, .loading, .loader, [class*="spinner" i], [aria-busy="true"]';
        var els = document.querySelectorAll(sel);
        for (var j = 0; j < els.length; j++) {
            if (els[j].getClientRects().length) { busy = true; break; }
        }
    } catch (e) {}
    return {state: document.readyState, sig: location.href + '#' + t.length + ':' + h, busy: busy};
})()
"""

    # Delay before the single consent re-check when the first pass matched
    # nothing (consent SDKs mount asynchronously after page readiness).
    _COOKIE_RETRY_DELAY = 1.0

    # Login/auth pages must NEVER get the consent pass: their "Continue" / "Sign
    # in" / SSO buttons look nothing like cookie banners yet a stray click submits
    # the auth flow (see the phantom-click rate-limit incident on x.com). Skip the
    # whole pass when the URL is part of an auth flow.
    _AUTH_URL_RE = re.compile(
        r'(/i/flow/|/i/jf/|/login|/log-in|/signin|/sign-in|/signup|/sign-up|'
        r'/oauth|/authorize|/auth/|accounts\.google\.|appleid\.apple\.|'
        r'login\.live\.|/sso|/session/new)',
        re.IGNORECASE,
    )

    # Class-level defaults so partially-constructed instances (tests use
    # WebAccess.__new__) read sane values; writes always land on the instance.
    _browse_fail_streak: int = 0
    _browser_hard_down: bool = False
    _browser_closed_by_user: bool = False
    # Incremented on every successful Chrome launch; a browse session records the
    # generation it was opened under so mid-session commands can detect a restart.
    _browser_generation: int = 0
    _browse_session_generation: int = 0
    # Iframe targets still without a URL at the last _async_list_frame_targets
    # (fresh OOPIFs Chrome has created but not committed) — the snapshot waits
    # for these. Frame labels / target ids the settle wait already gave up on
    # (an iframe src that redirects never matches its target's URL) are not
    # waited for again in this session. Both are reassigned, never mutated, so
    # these class-level defaults stay untouched.
    _blank_frame_target_ids: frozenset = frozenset()
    _oopif_settle_giveups: frozenset = frozenset()

    _PROFILE_DIR = str(CACHE_DIR / 'chrome_profile')

    def __init__(self):
        Integration.__init__(self)
        self._browser: Optional[nodriver.Browser] = None
        self._tab: Optional[nodriver.Tab] = None
        self._ua_cache: Optional[tuple[str, str, str]] = None
        self._chrome_pids: set[int] = set()
        self._profile_dir = os.path.normpath(self._PROFILE_DIR)
        os.makedirs(self._profile_dir, exist_ok=True)
        self._driver_ready = threading.Event()
        self._last_init_failure_ts: float = 0.0
        self._init_in_progress: bool = False
        self._browse_session_url: Optional[str] = None
        # Per snapshot index: owning iframe CDP target id, or None for the main page.
        self._browse_snapshot_frame_ids: list = []
        self._browse_snapshot_labels: list[str] = []  # parallel labels for human-readable describe
        self._browse_frame_sessions: dict = {}  # iframe target id -> attached devtools session id
        self._blank_frame_target_ids = frozenset()
        self._oopif_settle_giveups = frozenset()
        # Popup/new-tab tracking: parent tabs beneath the active popup (base tab at
        # index 0), page-target ids already accounted for, and notes queued for the
        # next snapshot when the active target changes.
        self._popup_parents: list = []
        self._known_page_targets: set = set()
        self._pending_target_notes: list[str] = []
        # JS dialogs (alert/confirm/prompt) auto-answered since the last browse
        # command replied; each is reported in that command's reply.
        self._dialog_notes: list[str] = []
        # Fresh page targets still on about:blank, tid -> first-seen timestamp;
        # adopted only once they navigate to a real URL.
        self._pending_popup_targets: dict[str, float] = {}
        # Network capture, oldest first; the dict maps an in-flight CDP request id
        # to its entry so the response / body / failure events find it.
        self._network_log: list[dict] = []
        self._network_requests: dict[str, dict] = {}
        self._browse_pdf_text: Optional[str] = None  # cached extracted text when session is a PDF
        self._page_texts: dict[str, str] = {}  # the long texts behind the last paged results
        self._last_screenshot_bytes: Optional[bytes] = None
        self._last_screenshot_path: Optional[str] = None
        self._browse_fail_streak = 0
        self._browser_hard_down = False
        self._browser_closed_by_user = False
        self._browser_generation = 0
        self._browse_session_generation = 0
        # Serializes browse-session commands on the shared tab (binds to the loop
        # at first await). Without it, a watchdog-abandoned command interleaved
        # with its successor on the same tab and wedged both.
        self._browse_lock = asyncio.Lock()
        self._loop_thread = _LoopThread()
        self._loop_thread.start()
        # Kick off Chrome initialization on the loop thread without blocking startup
        if self._loop_thread.is_running():
            asyncio.run_coroutine_threadsafe(
                self._async_init_with_signal(), self._loop_thread.loop
            )
        else:
            note("WebAccess loop thread failed to start — web scraping unavailable")
            self._driver_ready.set()
        atexit.register(self._shutdown_driver)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def _async_init_with_signal(self):
        try:
            await self._async_initialize_driver()
        except Exception as e:
            note(f"Chrome driver init failed — web scraping unavailable: {e}")
            self._browser = None
            self._tab = None
            self._last_init_failure_ts = time.time()
        finally:
            self._driver_ready.set()

    async def _async_maybe_retry_init(self):
        """Rate-limited recovery: re-attempt nodriver init when the browser is None.

        Called from the entry guards in `_async_fetch_via_chrome` and
        `_async_browse_open` so that a dead browser self-heals on the next
        fetch instead of staying down until process restart. Honors
        `_INIT_RETRY_BACKOFF` so a permanently broken Chrome doesn't get
        hammered, and uses `_init_in_progress` to serialize concurrent
        callers on the loop thread (safe without a lock since the check-and-
        set is synchronous within the coroutine before the first await).
        """
        if self._browser is not None:
            return
        if self._init_in_progress:
            return
        elapsed = time.time() - self._last_init_failure_ts
        if elapsed < self._INIT_RETRY_BACKOFF:
            trace(
                f"Skipping browser recovery — last failure was {elapsed:.0f}s ago, "
                f"backoff is {self._INIT_RETRY_BACKOFF:.0f}s"
            )
            return
        self._init_in_progress = True
        try:
            note("Browser is down — attempting recovery init")
            try:
                await self._async_initialize_driver()
            except Exception as e:
                # An escaping exception here used to blow up the calling browse
                # command with a bare "ERROR: failed" AND skip the failure
                # bookkeeping, so the next command hammered init again.
                note(f"Browser recovery init raised {type(e).__name__}: {e} — "
                     "will retry after backoff")
                self._browser = None
                self._tab = None
                self._last_init_failure_ts = time.time()
                return
            if self._browser is None:
                note("Browser recovery init failed — will retry after backoff")
            else:
                info("Browser recovery init succeeded")
        finally:
            self._init_in_progress = False

    @staticmethod
    async def _cdp_wait_for(coro, timeout: float):
        """asyncio.wait_for for CDP calls whose timeout the session must survive.

        Plain wait_for cancels nodriver's in-flight Transaction while it stays
        registered in the connection mapper; the late CDP reply then resolves a
        cancelled future inside the websocket listener task, which kills the
        listener and hangs every later call on that connection. Shielding
        leaves the call running to consume its own reply; its late
        result/error is retrieved and discarded."""
        task = asyncio.ensure_future(coro)
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            task.add_done_callback(lambda t: t.cancelled() or t.exception())
            raise

    @classmethod
    def _describe_error(cls, e: BaseException) -> str:
        """'TimeoutError: no reply within 10s' rather than the '' str() of a
        py3.12 TimeoutError — the empty 'click failed: ' the 2026-09-11 log showed."""
        if isinstance(e, asyncio.TimeoutError):
            return (f"TimeoutError: the page did not answer within {cls._EVAL_TIMEOUT:.0f}s "
                    "(its main thread is busy or blocked)")
        text = str(e)
        return f"{type(e).__name__}: {text}" if text else type(e).__name__

    async def _async_browser_alive(self, timeout: float = _BROWSER_ALIVE_PROBE_TIMEOUT) -> bool:
        """Cheap liveness probe: round-trip Target.getTargets on the browser-level CDP connection.

        Uses the browser connection, not the tab — the tab may be a dead popup
        target while Chrome itself is fine."""
        if self._browser is None:
            return False
        try:
            await asyncio.wait_for(self._browser.connection.send(cdp.target.get_targets()), timeout)
            return True
        except Exception as e:
            note(f"Browser liveness probe failed ({type(e).__name__}: {e}) — treating browser as dead")
            return False

    async def _async_ensure_browser(self, force: bool = False) -> bool:
        """Single reinit funnel: verify Chrome is alive and restart it if not.

        Respects the init-failure backoff unless force=True (circuit breaker).
        Returns True when a usable tab exists after the check."""
        await self._async_wait_for_driver()
        if await self._async_browser_alive():
            if self._tab is not None:
                return True
            # _adopt_newest_live_target found nothing adoptable and cleared the
            # tab: Chrome is fine but there is no page left to drive. Restart it
            # so the generation bump routes mid-session commands into the
            # auto-reopen path instead of failing on the missing tab forever.
            note("browse: browser is alive but no tab is left to drive — restarting it")
            await self._async_quit_driver()
        elif self._browser is not None:
            note("Browser is unresponsive — shutting it down before reinit")
            await self._async_quit_driver()
        if force:
            self._last_init_failure_ts = 0.0
        await self._async_maybe_retry_init()
        return self._tab is not None

    def _mark_browser_up(self):
        """Reset failure bookkeeping after a successful Chrome launch."""
        self._browser_generation += 1
        self._browse_fail_streak = 0
        self._browser_hard_down = False
        self._browser_closed_by_user = False

    async def _async_infra_error(self, msg: str) -> str:
        """Register an infra-class browse failure and build its ERROR reply.

        At _BREAKER_TRIP_THRESHOLD consecutive failures, force one restart past
        the backoff; if even that fails, latch the hard-down state so browse
        commands fail fast until a later successful init clears it."""
        self._browse_fail_streak += 1
        note(f"browse: infra failure #{self._browse_fail_streak}: {msg}")
        if self._browse_fail_streak >= self._BREAKER_TRIP_THRESHOLD and not self._browser_hard_down:
            note("browse: fail streak tripped the circuit breaker — forcing one restart past backoff")
            if await self._async_ensure_browser(force=True):
                info("browse: forced restart succeeded — resetting fail streak")
                self._browse_fail_streak = 0
            else:
                note("browse: forced restart failed too — latching browser hard-down")
                self._browser_hard_down = True
                return self._HARD_DOWN_ERROR
        return f"ERROR: {msg}"

    async def _async_browse_ready(self) -> Optional[str]:
        """Health gate run at the top of every browse command.

        Returns None when the browser is up (restarting it transparently if
        needed), or the final ERROR string the command should return as-is."""
        if self._browser_hard_down:
            trace("browse: hard-down latch set — failing fast without probe")
            return self._HARD_DOWN_ERROR
        if await self._async_ensure_browser():
            return None
        err = await self._async_infra_error("browser is not available on this server.")
        if self._browse_fail_streak == 0 and self._tab is not None:
            info("browse: breaker-forced restart recovered the browser — continuing with the command")
            return None
        return err

    async def _async_browse_ready_midsession(self) -> Optional[str]:
        """Health gate for commands acting on an existing (non-PDF) browse session.

        If the browser was restarted since browse_open, the page and snapshot died
        with it — auto-reopen the session URL and return a NOTE + fresh snapshot
        instead of letting the command execute against a stale index."""
        if err := await self._async_browse_ready():
            return err
        if self._browse_session_generation == self._browser_generation:
            return None
        url = self._browse_session_url
        note(f"browse: browser was restarted mid-session — reopening {url}")
        result = await self._async_browse_open(url)
        if isinstance(result, str) and result.startswith("ERROR"):
            return result
        return (f"NOTE: the browser was restarted and {url} was reloaded — element indices "
                f"changed, here is a fresh snapshot:\n\n{result}")

    def _shutdown_driver(self):
        """Stop the browser, force-kill any orphan Chrome PIDs, then stop the loop thread."""
        if self._browser is not None:
            try:
                self._track_chrome_pids()
                self._browser.stop()
            except Exception:
                pass
            self._browser = None
            self._tab = None
        if self._loop_thread.is_running():
            try:
                self._loop_thread.stop()
            except Exception:
                pass
        self._kill_tracked_chrome_pids()
        # Final sweep: Chrome helper children can outlive even a SIGKILL of the
        # browser process; anything still holding our profile dir is ours to kill.
        try:
            self._kill_orphan_chrome_using_profile()
        except Exception as e:
            trace(f"Final orphan-Chrome profile sweep failed: {e}")

    def _track_chrome_pids(self):
        """Record the PID of the Chrome process spawned by the current browser."""
        if not hasattr(self, '_chrome_pids'):
            self._chrome_pids = set()
        if self._browser is None:
            return
        try:
            proc = getattr(self._browser, '_process', None)
            if proc is not None and getattr(proc, 'pid', None):
                self._chrome_pids.add(proc.pid)
            pid = getattr(self._browser, '_process_pid', None)
            if isinstance(pid, int) and pid > 0:
                self._chrome_pids.add(pid)
        except Exception:
            pass

    @staticmethod
    def _chrome_child_processes(pid: int) -> list:
        """Live child processes (recursive) of the given PID, or [] if it's gone."""
        try:
            return psutil.Process(pid).children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return []
        except Exception:
            return []

    def _kill_tracked_chrome_pids(self):
        """Force-kill any Chrome PIDs we previously tracked, including their helper children.

        Chrome helpers (GPU/renderer/utility) can survive a SIGKILL of the browser
        process itself — the observed orphan accumulation — so collect the child
        tree via psutil before killing the parent."""
        if not hasattr(self, '_chrome_pids'):
            return
        for pid in list(self._chrome_pids):
            children = self._chrome_child_processes(pid)
            for child in children:
                try:
                    child.kill()
                    info(f"Force-killed Chrome helper process {child.pid} (child of {pid})")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                except Exception:
                    pass
            try:
                os.kill(pid, signal.SIGKILL)
                info(f"Force-killed orphaned Chrome process {pid}")
            except ProcessLookupError:
                pass
            except Exception:
                pass
        self._chrome_pids.clear()

    def _kill_orphan_chrome_using_profile(self):
        """Kill any leftover Chrome process from a previous run that still owns our profile.

        Singleton-lock cleanup only removes lock *files*. If a prior server run leaked a
        live Chrome process, it keeps the profile bound at the OS/IPC level and the next
        nodriver.start() fails with 'Failed to connect to browser'. Sweep those orphans
        before launch.
        """
        profile_marker = f"--user-data-dir={self._profile_dir}"
        own_pid = os.getpid()
        victims: list[psutil.Process] = []
        try:
            # OSError included: on macOS, reading the cmdline of a process that is
            # mid-death (e.g. one browse_close just SIGKILLed) raises
            # OSError [Errno 0] from sysctl(KERN_PROCARGS2) — it must not abort
            # the sweep, let alone the browser init that called it.
            for proc in psutil.process_iter(['pid', 'cmdline']):
                try:
                    if proc.info['pid'] == own_pid:
                        continue
                    cmdline = proc.info.get('cmdline') or []
                    if any(profile_marker in arg for arg in cmdline):
                        victims.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                    continue
        except (psutil.Error, OSError) as e:
            note(f"Orphan Chrome sweep stopped mid-scan ({type(e).__name__}: {e}) — "
                 f"continuing with {len(victims)} candidate(s) found so far")
        if not victims:
            return
        info(f"Found {len(victims)} orphan Chrome process(es) holding {self._profile_dir} — terminating")
        for proc in victims:
            try:
                proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                pass
        _, alive = psutil.wait_procs(victims, timeout=2)
        for proc in alive:
            try:
                proc.kill()
                info(f"Force-killed stubborn orphan Chrome PID {proc.pid}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                pass

    def _clear_singleton_locks(self):
        """Remove stale Chrome singleton lock files from the persistent profile.

        Chrome refuses to reuse a profile if Singleton{Lock,Cookie,Socket} exist
        from a prior crashed run. WebAccess is the sole owner of this profile,
        so clearing them on startup is safe.
        """
        cleared = False
        for name in ('SingletonLock', 'SingletonCookie', 'SingletonSocket'):
            path = os.path.join(self._profile_dir, name)
            if os.path.lexists(path):
                try:
                    os.unlink(path)
                    cleared = True
                except OSError as e:
                    trace(f"Failed to remove stale lock {path}: {e}")
        if cleared:
            info(f"Cleared stale Chrome singleton locks in {self._profile_dir}")

    def _wipe_profile(self):
        """Delete and recreate the profile directory after an init failure."""
        try:
            shutil.rmtree(self._profile_dir, ignore_errors=True)
        finally:
            os.makedirs(self._profile_dir, exist_ok=True)
        trace(f"Wiped Chrome profile at {self._profile_dir}")

    _CHROME_PATH_NAMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")

    @staticmethod
    def _chrome_search_paths() -> list[str]:
        """The explicit binary paths probed when locating Chrome/Chromium."""
        candidates = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
        ]
        if platform.system() == "Linux":
            candidates += [
                "/usr/bin/google-chrome",
                "/usr/bin/google-chrome-stable",
                "/usr/bin/chromium",
                "/usr/bin/chromium-browser",
            ]
        return candidates

    @classmethod
    def _find_chrome_binary(cls) -> Optional[str]:
        """Locate the Chrome/Chromium binary on this system."""
        for path in cls._chrome_search_paths():
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        for name in cls._CHROME_PATH_NAMES:
            found = shutil.which(name)
            if found:
                return found
        return None

    async def _async_initialize_driver(self):
        """Launch a fresh nodriver browser, register stealth + network monitor scripts."""
        await self._async_quit_driver()
        self._ua_cache = None             # re-probe UA; a Chrome update may bump the version
        self._browse_frame_sessions = {}  # sessions belonged to the old connection
        self._popup_parents = []          # popup tabs belonged to the old browser
        self._known_page_targets = set()
        self._pending_target_notes = []
        self._dialog_notes = []
        self._pending_popup_targets = {}

        chrome_path = self._find_chrome_binary()
        if not chrome_path:
            install_hint = (
                "install it with 'brew install --cask google-chrome'"
                if platform.system() == "Darwin"
                else "install Google Chrome or Chromium (e.g. 'apt install chromium')"
            )
            error(
                "Web Access disabled: no Chrome/Chromium binary found. "
                "nodriver needs a Chrome/Chromium install to scrape web pages — "
                f"{install_hint}; the next web command picks it up. "
                f"Searched: {', '.join(self._chrome_search_paths())}"
            )
            self._browser = None
            self._tab = None
            return

        trace(f"Using Chrome binary: {chrome_path}")
        trace(f"Using persistent Chrome profile: {self._profile_dir}")
        # The UA goes on the command line, not through Emulation.setUserAgentOverride:
        # with the CDP override Chrome emits client-hint headers that no real Chrome
        # sends (a hand-built brand list, or none at all), and DataDome hard-blocks
        # on that from the very first request (Etsy, 2026-09-29: t=bv with the
        # override, a solvable challenge without it). With the flag Chrome derives
        # the Sec-Ch-Ua headers itself and popups inherit the UA. No --disable-gpu
        # either: SwiftShader as the WebGL renderer is another headless tell.
        ua = await asyncio.to_thread(self._launch_user_agent, chrome_path)
        # StrictOriginIsolation: Chrome's default site isolation splits frames by
        # site (scheme + registrable domain), so a cross-origin iframe on a sub-domain
        # of the page's own site (Tesla's static-assets-pay.tesla.com inside
        # shop.tesla.com, app.bokabord.se inside www.bokabord.se) stays in the page's
        # renderer: no CDP target, and contentDocument is blocked — the snapshot can
        # read it from neither side. Isolating by origin makes every cross-origin
        # frame an out-of-process one, which the snapshot already collects from.
        browser_args = [
            f'--user-agent={ua}',
            '--window-size=1920,1080',
            '--disable-dev-shm-usage',
            '--enable-features=StrictOriginIsolation',
        ]

        self._kill_orphan_chrome_using_profile()
        self._clear_singleton_locks()

        async def _start():
            return await nodriver.start(
                headless=True,
                browser_executable_path=chrome_path,
                browser_args=browser_args,
                sandbox=False,
                user_data_dir=self._profile_dir,
            )

        try:
            self._browser = await _start()
        except Exception as e:
            trace(
                f"nodriver.start() failed: {e} — sleeping "
                f"{self._INIT_INTER_RETRY_DELAY:.0f}s for Chrome to release resources, "
                f"then retrying"
            )
            self._browser = None
            self._tab = None
            # Give the previous Chrome process tree time to die. Both retries
            # used to happen within ~3s of each other, which is too tight on macOS;
            # child renderers and the OS often need 10–15s to fully release.
            await asyncio.sleep(self._INIT_INTER_RETRY_DELAY)
            self._kill_orphan_chrome_using_profile()
            self._clear_singleton_locks()
            try:
                self._browser = await _start()
            except Exception as e2:
                # Only now is the profile a suspect: a start that fails after the
                # process tree had time to die is not a dying-Chrome collision.
                # The profile holds every login cookie, so it is the last resort.
                note(f"nodriver.start() failed again ({e2}) — wiping the profile and retrying once more")
                self._browser = None
                self._tab = None
                self._wipe_profile()
                self._kill_orphan_chrome_using_profile()
                self._clear_singleton_locks()
                try:
                    self._browser = await _start()
                except Exception as e3:
                    note(f"nodriver.start() failed after profile wipe: {e3}")
                    self._browser = None
                    self._tab = None
                    self._last_init_failure_ts = time.time()
                    return

        self._track_chrome_pids()

        try:
            self._tab = await self._browser.get("about:blank")
            # Page domain must be enabled before addScriptToEvaluateOnNewDocument takes effect
            try:
                await self._tab.send(cdp.page.enable())
            except Exception as e:
                trace(f"page.enable() failed: {e}")
            self._install_dialog_handler(self._tab)
            self._install_network_capture(self._tab)
            await self._async_install_stealth_script(self._tab)
            await self._async_setup_network_monitor(self._tab)
            info(f"nodriver initialized (UA {ua})")
            self._last_init_failure_ts = 0.0
            self._mark_browser_up()
            # Seed the popup bookkeeping with every page already open so the
            # first sync after (re)init doesn't mistake the base tab or leftover
            # blank targets for freshly opened popups and adopt them.
            self._known_page_targets = {str(self._tab.target.target_id)} | {
                str(t.target.target_id) for t in self._page_targets()
            }
            trace(f"popup: seeded {len(self._known_page_targets)} known page target(s) after init")
        except Exception as e:
            note(f"nodriver post-start setup failed: {e}")
            try:
                self._browser.stop()
            except Exception:
                pass
            self._browser = None
            self._tab = None
            self._last_init_failure_ts = time.time()

    # How long _async_quit_driver waits for the killed Chrome tree to actually exit.
    # Chrome's helpers take seconds to die on macOS; a browse_open right after
    # browse_close used to launch into that window, fail with "Failed to connect
    # to browser" and (before the retry was reordered) wipe the profile.
    _CHROME_EXIT_TIMEOUT = 10.0

    async def _async_quit_driver(self):
        """Close the current browser, force-kill its PIDs if stop fails, and wait
        until the whole process tree has exited so the next launch does not
        collide with it."""
        if self._browser is None:
            return
        self._track_chrome_pids()
        procs = self._tracked_chrome_processes()
        try:
            self._browser.stop()
        except Exception:
            pass
        self._browser = None
        self._tab = None
        self._kill_tracked_chrome_pids()
        await self._async_wait_for_chrome_exit(procs)

    def _tracked_chrome_processes(self) -> list:
        """psutil handles for every tracked Chrome PID and its helper children —
        collected before the kill, since killing clears the tracked set."""
        procs = []
        for pid in list(getattr(self, '_chrome_pids', ())):
            try:
                procs.append(psutil.Process(pid))
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                continue
            procs.extend(self._chrome_child_processes(pid))
        return procs

    async def _async_wait_for_chrome_exit(self, procs: list) -> None:
        if not procs:
            return
        started = time.time()
        try:
            _gone, alive = await asyncio.to_thread(psutil.wait_procs, procs, self._CHROME_EXIT_TIMEOUT)
        except Exception as e:
            trace(f"browse: waiting for the Chrome tree to exit failed: {type(e).__name__}: {e}")
            return
        if alive:
            note(f"browse: {len(alive)} of {len(procs)} Chrome process(es) still alive "
                 f"{self._CHROME_EXIT_TIMEOUT:.0f}s after the kill: {[p.pid for p in alive]}")
        else:
            info(f"browse: Chrome process tree ({len(procs)} process(es)) exited in {time.time() - started:.1f}s")

    # Fallback Chrome major version for the launch UA when `chrome --version`
    # cannot be read — keep roughly current so the fingerprint isn't obviously
    # stale. The live value is preferred and used when available.
    _UA_FALLBACK_MAJOR = "154"
    _CHROME_VERSION_PROBE_TIMEOUT = 5.0

    @staticmethod
    def _ua_platform_token() -> str:
        system = platform.system()
        if system == "Darwin":
            return "Macintosh; Intel Mac OS X 10_15_7"
        if system == "Windows":
            return "Windows NT 10.0; Win64; x64"
        return "X11; Linux x86_64"

    def _launch_user_agent(self, chrome_path: str) -> str:
        """The user agent passed to Chrome on its command line: what this Chrome
        version sends when it is not headless. Chrome reduces the UA to
        `Chrome/<major>.0.0.0` itself, so the major from `chrome --version` is all
        it takes. Cached per launch; a Chrome update is picked up on the next one."""
        cached = getattr(self, '_ua_cache', None)
        if cached:
            return cached
        major = None
        try:
            out = subprocess.run([chrome_path, '--version'], capture_output=True, text=True,
                                 timeout=self._CHROME_VERSION_PROBE_TIMEOUT).stdout
            m = re.search(r'\b(\d+)\.\d+\.\d+\.\d+', out or '')
            major = m.group(1) if m else None
            if major is None:
                trace(f"chrome --version gave no version: {out!r} — using fallback major")
        except Exception as e:
            trace(f"chrome --version failed: {type(e).__name__}: {e} — using fallback major")
        major = major or self._UA_FALLBACK_MAJOR
        self._ua_cache = (f"Mozilla/5.0 ({self._ua_platform_token()}) AppleWebKit/537.36 "
                          f"(KHTML, like Gecko) Chrome/{major}.0.0.0 Safari/537.36")
        return self._ua_cache

    # The only fingerprint patch left. Headless Chrome answers a notifications
    # permission query with 'denied' while Notification.permission says
    # 'default' — the mismatch is a classic headless tell. nodriver already
    # removes navigator.webdriver, the chrome.runtime shim and the cdc_* patches.
    # No WebGL spoof: the 'Intel Inc. / Intel Iris OpenGL Engine' pair is the
    # puppeteer-extra-stealth signature every bot vendor fingerprints.
    # A Proxy, not a wrapper function: Function.prototype.toString of a proxied native
    # still reads `[native code]` and `.name` is still 'query', so the patch itself is
    # not the tell it replaces.
    _STEALTH_JS = """
        navigator.permissions.query = new Proxy(navigator.permissions.query, {
            apply: (target, thisArg, args) =>
                args[0] && args[0].name === 'notifications'
                    ? Promise.resolve({state: Notification.permission, onchange: null})
                    : Reflect.apply(target, thisArg, args)
        });
    """

    async def _async_install_stealth_script(self, tab: 'nodriver.Tab'):
        try:
            await tab.send(cdp.page.add_script_to_evaluate_on_new_document(source=self._STEALTH_JS))
        except Exception as e:
            trace(f"stealth script registration failed: {e}")

    # DOM-quiet signal for _async_wait_for_page_ready: __domMon.last is the
    # timestamp of the newest DOM mutation. MutationObserver already batches
    # mutations per microtask checkpoint, so the callback is cheap enough to
    # just stamp Date.now(). documentElement may not exist yet when this runs
    # (script-on-new-document fires at document creation), hence the re-poll.
    _DOM_MONITOR_JS = """
(function() {
    if (window.__domMon) return;
    window.__domMon = { last: Date.now() };
    var mon = window.__domMon;
    function attach() {
        if (document.documentElement) {
            new MutationObserver(function() { mon.last = Date.now(); })
                .observe(document.documentElement,
                         { childList: true, subtree: true, attributes: true });
        } else {
            setTimeout(attach, 10);
        }
    }
    attach();
})();
"""

    def _install_dialog_handler(self, tab: 'nodriver.Tab') -> None:
        """Auto-answer every JavaScript dialog the tab opens.

        Under --headless=new a native alert/confirm/prompt is not auto-dismissed:
        it parks the renderer main thread until a CDP client answers with
        Page.handleJavaScriptDialog, and every later evaluate/click on the tab
        times out behind it (the 2026-09-11 BokaBord "Ta bort bokning" hang —
        a confirm() in an ng-click). Accepting is what a user clicking through
        the flow would do; the dialog text is queued for the command's reply so
        the assistant sees the question it answered. Page must be enabled on
        the tab for the event to arrive."""
        try:
            tab.add_handler(cdp.page.JavascriptDialogOpening, self._on_javascript_dialog)
        except Exception as e:
            trace(f"dialog handler registration failed: {type(e).__name__}: {e}")

    async def _on_javascript_dialog(self, event: 'cdp.page.JavascriptDialogOpening',
                                    connection=None) -> None:
        kind = getattr(event, 'type_', None)
        kind = str(getattr(kind, 'value', kind) or 'dialog')  # DialogType enum -> 'confirm'
        message = (getattr(event, 'message', '') or '').strip()
        # A prompt() gets its own default text back; the others just need "OK".
        prompt_text = getattr(event, 'default_prompt', None) if kind == 'prompt' else None
        tab = connection if connection is not None else self._tab
        info(f"browse: auto-accepting {kind} dialog: {message!r}")
        try:
            await tab.send(cdp.page.handle_java_script_dialog(accept=True, prompt_text=prompt_text))
        except Exception as e:
            note(f"browse: accepting {kind} dialog failed: {type(e).__name__}: {e}")
            return
        self._dialog_notes.append(
            f"NOTE: the page showed a {kind} dialog \u201c{message}\u201d which was accepted (OK) "
            "automatically.")

    def _take_dialog_notes(self) -> str:
        """Drain the dialog notes as a reply prefix ('' when none)."""
        if not self._dialog_notes:
            return ''
        notes, self._dialog_notes = self._dialog_notes, []
        return "\n".join(notes) + "\n\n"

    async def _async_setup_network_monitor(self, tab: 'nodriver.Tab'):
        """Inject JS via CDP to track DOM mutations on every new page."""
        try:
            await tab.send(cdp.page.add_script_to_evaluate_on_new_document(source=self._DOM_MONITOR_JS))
        except Exception as e:
            trace(f"DOM mutation monitor setup failed: {e}")

    # ------------------------------------------------------------------
    # Network capture
    # ------------------------------------------------------------------

    def _install_network_capture(self, tab: 'nodriver.Tab') -> None:
        """Record the tab's XHR/fetch/document traffic for browse_network().

        nodriver enables the Network domain on the tab as soon as a handler for
        one of its events is registered, so this is all it takes. Installed on
        the base tab and on every adopted popup — a payment popup's own polling
        (Klarna asking "is the BankID order complete?") is what tells a rejected
        signing from one that never happened."""
        try:
            tab.add_handler(cdp.network.RequestWillBeSent, self._on_network_request)
            tab.add_handler(cdp.network.ResponseReceived, self._on_network_response)
            tab.add_handler(cdp.network.LoadingFinished, self._on_network_finished)
            tab.add_handler(cdp.network.LoadingFailed, self._on_network_failed)
        except Exception as e:
            trace(f"network capture registration failed: {type(e).__name__}: {e}")

    @staticmethod
    def _network_type(event) -> str:
        kind = getattr(event, 'type_', None)
        return str(getattr(kind, 'value', kind) or '')

    @staticmethod
    def _network_tab_id(connection) -> str:
        try:
            return str(connection.target.target_id)[:8]
        except Exception:
            return '?'

    def _network_touch(self) -> None:
        self._network_last_activity = time.time()

    def _network_in_flight(self) -> list:
        """Captured requests still unanswered — XHR/fetch/document, not EventSource (a
        stream never finishes) and not those older than _PAGE_NET_STALE_S."""
        horizon = time.time() - self._PAGE_NET_STALE_S
        return [e for e in getattr(self, '_network_requests', {}).values()
                if not e['done'] and e['type'] != 'EventSource' and e['ts'] >= horizon]

    def _network_failures_since(self, since: Optional[float]) -> list[tuple[int, dict]]:
        """The captured requests sent at or after `since` that the server refused (4xx/5xx)
        or that never got an answer, with their browse_network indices. None → nothing."""
        if since is None:
            return []
        return [(i, e) for i, e in enumerate(getattr(self, '_network_log', None) or [])
                if e['ts'] >= since and ((e['status'] or 0) >= 400 or e['note'].startswith('failed'))]

    def _network_add(self, entry: dict) -> None:
        self._network_touch()
        self._network_log.append(entry)
        self._network_requests[entry['rid']] = entry
        while len(self._network_log) > self._NETWORK_LOG_MAX:
            old = self._network_log.pop(0)
            if self._network_requests.get(old['rid']) is old:
                del self._network_requests[old['rid']]

    @classmethod
    def _scrub_request_body(cls, post: str) -> str:
        """A request body safe to keep, trace and hand to the LLM.

        Credential-looking fields (password, otp, card number…) are masked by name, and the
        body is decoded into the form the vault redactor matches on: a form POST arrives
        percent-encoded (`p%40ss+w0rd`) and a JSON one may carry `\\u00e5` escapes, neither of
        which the literal-match redactor recognises as the stored `p@ss w0rd`. A body that is
        neither is kept as it came."""
        body = post.strip()
        if not body:
            return ''
        if body[0] in '{[':
            try:
                data = json.loads(body)
            except ValueError:
                return post
            scrubbed = cls._scrub_json(data)
            if scrubbed == data and '\\u' not in body:
                return post
            return json.dumps(scrubbed, ensure_ascii=False, separators=(',', ':'))
        if cls._NETWORK_FORM_BODY_RE.match(body):
            pairs = parse_qsl(body, keep_blank_values=True, errors='replace')
            if pairs:
                return '&'.join(f"{k}={cls._NETWORK_SECRET_MASK if cls._NETWORK_SECRET_KEY_RE.search(k) else v}"
                                for k, v in pairs)
        return post

    @classmethod
    def _scrub_json(cls, data):
        if isinstance(data, dict):
            return {k: cls._NETWORK_SECRET_MASK if isinstance(k, str) and cls._NETWORK_SECRET_KEY_RE.search(k)
                    else cls._scrub_json(v) for k, v in data.items()}
        if isinstance(data, list):
            return [cls._scrub_json(v) for v in data]
        return data

    def _on_network_request(self, event: 'cdp.network.RequestWillBeSent', connection=None) -> None:
        kind = self._network_type(event)
        if kind not in self._NETWORK_CAPTURE_TYPES:
            return
        if self._NETWORK_NOISE_URL_RE.match(event.request.url):
            fine(f"network: beacon not captured — {event.request.method} {event.request.url[:80]}")
            return
        rid = str(event.request_id)
        redirect = getattr(event, 'redirect_response', None)
        if redirect is not None and (prev := self._network_requests.get(rid)):
            # Same request id continues on the new URL: close the hop it came from.
            prev['status'] = getattr(redirect, 'status', None)
            prev['mime'] = getattr(redirect, 'mime_type', '') or ''
            prev['note'] = f"redirected to {event.request.url}"
            prev['done'] = True
        post = self._scrub_request_body(getattr(event.request, 'post_data', None) or '')
        entry = {
            'rid': rid, 'ts': time.time(), 'tab': self._network_tab_id(connection), 'type': kind,
            'method': event.request.method, 'url': event.request.url,
            'post': post[:self._NETWORK_BODY_MAX], 'status': None, 'mime': '', 'size': None,
            'body': None, 'note': '', 'done': False,
        }
        self._network_add(entry)
        trace(f"network[{entry['tab']}]: → {kind} {entry['method']} {entry['url']}"
              + (f" body {post[:self._NETWORK_BODY_TRACE_MAX]!r}" if post else ''))

    def _on_network_response(self, event: 'cdp.network.ResponseReceived', connection=None) -> None:
        entry = self._network_requests.get(str(event.request_id))
        if entry is None:
            return
        self._network_touch()
        entry['status'] = event.response.status
        entry['mime'] = event.response.mime_type or ''
        trace(f"network[{entry['tab']}]: ← {entry['type']} {entry['status']} {entry['method']} "
              f"{entry['url']} ({entry['mime']})")

    async def _on_network_finished(self, event: 'cdp.network.LoadingFinished', connection=None) -> None:
        entry = self._network_requests.get(str(event.request_id))
        if entry is None:
            return
        self._network_touch()
        entry['done'] = True
        entry['size'] = int(getattr(event, 'encoded_data_length', 0) or 0)
        if entry['type'] not in self._NETWORK_BODY_TYPES or not self._NETWORK_BODY_MIME_RE.search(entry['mime']):
            return
        if connection is None:
            return
        try:
            body, b64 = await self._cdp_wait_for(
                connection.send(cdp.network.get_response_body(request_id=event.request_id)),
                self._NETWORK_BODY_FETCH_TIMEOUT)
        except Exception as e:
            trace(f"network[{entry['tab']}]: body of {entry['url']} unavailable: {self._describe_error(e)}")
            return
        if b64:
            try:
                body = base64.b64decode(body).decode('utf-8', 'replace')
            except Exception:
                body = '<binary>'
        entry['body'] = (body or '')[:self._NETWORK_BODY_MAX]
        trace(f"network[{entry['tab']}]: {entry['status']} {entry['url']} body: "
              f"{(body or '')[:self._NETWORK_BODY_TRACE_MAX]!r}")

    def _on_network_failed(self, event: 'cdp.network.LoadingFailed', connection=None) -> None:
        entry = self._network_requests.get(str(event.request_id))
        if entry is None:
            return
        self._network_touch()
        entry['done'] = True
        entry['note'] = 'canceled' if getattr(event, 'canceled', False) else f"failed: {event.error_text}"
        trace(f"network[{entry['tab']}]: ✗ {entry['method']} {entry['url']} — {entry['note']}")

    def _clear_network_log(self) -> None:
        # getattr: partially-constructed instances (tests use WebAccess.__new__) reach here too.
        if getattr(self, '_network_log', None):
            trace(f"network: dropping {len(self._network_log)} captured entries of the previous session")
        self._network_log = []
        self._network_requests = {}
        self._network_last_activity = 0.0

    def _format_network_entry(self, i: int, e: dict) -> str:
        stamp = time.strftime('%H:%M:%S', time.localtime(e['ts']))
        status = e['status'] if e['status'] is not None else ('…' if not e['done'] else '-')
        size = f", {e['size']} B" if e['size'] else ''
        line = f"[{i}] {stamp} tab {e['tab']} {e['type']} {e['method']} {status} {e['url']} ({e['mime'] or '?'}{size})"
        if e['note']:
            line += f" — {e['note']}"
        if e['post']:
            line += f"\n    > {' '.join(e['post'].split())}"
        if e['body']:
            line += f"\n    < {' '.join(e['body'].split())}"
        return line

    async def _async_browse_network(self, url_filter: str) -> str:
        if not self._browse_session_url:
            return "ERROR: no browse session. Call browse_open(url) first."
        entries = list(enumerate(self._network_log))
        url_filter = (url_filter or '').strip()
        if url_filter:
            try:
                rx = re.compile(url_filter, re.I)
            except re.error:
                rx = re.compile(re.escape(url_filter), re.I)
            entries = [(i, e) for i, e in entries if rx.search(e['url'])]
        total = len(self._network_log)
        if not entries:
            info(f"browse_network: nothing captured{' matching ' + url_filter if url_filter else ''} "
                 f"({total} entries in the session)")
            return (f"No XHR/fetch/document traffic captured"
                    + (f" matching {url_filter!r} ({total} entries total)" if url_filter else '')
                    + " since browse_open.")
        shown = entries[-self._NETWORK_REPORT_DEFAULT:]
        lines = [self._format_network_entry(i, e) for i, e in shown]
        out = "\n".join(lines)
        if len(out) > self._NETWORK_REPORT_MAX_CHARS:
            out = "… (older entries cut)\n" + out[-self._NETWORK_REPORT_MAX_CHARS:]
        info(f"browse_network: {len(shown)} of {len(entries)} matching entries ({total} captured)")
        head = (f"{NETWORK_LOG_HEADER}{len(shown)} of {len(entries)} matching entries ({total} captured this "
                f"session), oldest first; `>` = request body, `<` = response body (excerpted):\n")
        return head + out

    def browse_network(self, url_filter: str = '') -> str:
        """Recent XHR/fetch/document responses of the browse session (all tabs), optionally
        filtered by a URL regex. Read-only: no snapshot, no page interaction."""
        info(f"browse_network{'(' + url_filter.strip() + ')' if url_filter and url_filter.strip() else ''}")
        return self._run_browse_op("browse_network", self._async_browse_network(url_filter),
                                   "ERROR: network log unavailable.")

    # ------------------------------------------------------------------
    # Cookie banner removal & navigation helpers
    # ------------------------------------------------------------------

    _COOKIE_REMOVAL_JS = r"""
(function() {
    let clicked = 0;
    let removed = 0;
    let kept = 0;

    // Nothing below may ever delete the document or the page's content: feature
    // classes land on <html> itself (Apple: "supports-backdrop-filter"), and layout
    // wrappers are routinely classed "has-banner" / "overlay-host".
    // Readable text only: textContent also counts <script>/<style> source, and a JSON
    // blob on a client-rendered page would swamp the ratio in either direction.
    const textLen = (el) => {
        if (!el) return 0;
        let len = (el.textContent || '').length;
        try {
            el.querySelectorAll('script, style, noscript').forEach(code => {
                len -= (code.textContent || '').length;
            });
        } catch (e) {}
        return Math.max(len, 0);
    };
    const isProtected = (el) => {
        if (!el || el === document.documentElement || el === document.body || el === document.head) return true;
        const tag = (el.tagName || '').toLowerCase();
        if (tag === 'html' || tag === 'body' || tag === 'head' || tag === 'main') return true;
        try {
            if (document.body && el.contains(document.body)) return true;
            if (el.querySelector('main, [role="main"], article')) return true;
        } catch (e) {}
        const total = textLen(document.body);
        return total > 0 && textLen(el) / total > 0.5;
    };
    const safeRemove = (el) => {
        if (isProtected(el)) { kept++; return false; }
        el.remove();
        removed++;
        return true;
    };
    // An overlay is positioned out of the flow AND covers the viewport, sits on a
    // raised layer or is a dialog — a class name alone proves nothing.
    const looksLikeOverlay = (el) => {
        const style = window.getComputedStyle(el);
        if (!['fixed', 'absolute', 'sticky'].includes(style.position)) return false;
        const role = (el.getAttribute && el.getAttribute('role')) || '';
        if (role === 'dialog' || role === 'alertdialog' ||
            (el.getAttribute && el.getAttribute('aria-modal') === 'true')) return true;
        if ((parseInt(style.zIndex) || 0) >= 100) return true;
        const rect = el.getBoundingClientRect();
        const viewport = (window.innerWidth || 0) * (window.innerHeight || 0);
        return viewport > 0 && (rect.width * rect.height) / viewport >= 0.5;
    };

    // Pass 0 — platform-specific accept buttons (accept-first: clicking persists
    // consent in cookies/localStorage so subsequent pages in the session are clean).
    const platformSelectors = [
        '#onetrust-accept-btn-handler',
        '#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll',
        '#CybotCookiebotDialogBodyButtonAccept',
        '#truste-consent-button',
        '#didomi-notice-agree-button',
        '.qc-cmp2-summary-buttons button[mode="primary"]',
        '.cmplz-accept',
        '.cky-btn-accept',
    ];
    platformSelectors.forEach(sel => {
        try {
            const el = document.querySelector(sel);
            if (el && el.offsetParent !== null) { el.click(); clicked++; }
        } catch (e) {}
    });
    // Usercentrics renders inside a shadow root — traverse it explicitly.
    try {
        const ucRoot = document.querySelector('#usercentrics-root');
        if (ucRoot && ucRoot.shadowRoot) {
            const btn = ucRoot.shadowRoot.querySelector('[data-testid="uc-accept-all-button"]');
            if (btn) { btn.click(); clicked++; }
        }
    } catch (e) {}
    // Usercentrics' newer CMP (web.cmp.usercentrics.eu — fedex.com, 2026-11): an
    // <aside id="usercentrics-cmp-ui"> shadow host whose overlay dims the whole page
    // and catches every real click until the wall is answered (a JS click passes
    // through it, which is how the old click "worked" there). Accept all when the
    // layer shows that button, else confirm the default selection.
    try {
        const cmp = document.querySelector('#usercentrics-cmp-ui');
        if (cmp && cmp.shadowRoot) {
            const btn = ['#accept', '.uc-accept-button', '#save', '.uc-save-button']
                .map(sel => cmp.shadowRoot.querySelector(sel))
                .find(b => b && (b.offsetWidth || b.offsetHeight));
            if (btn) { btn.click(); clicked++; }
        }
    } catch (e) {}

    // Pass 1 — click obvious accept buttons (collapsed from the old XPath loop)
    // Word boundaries + length cap prevent false-positive clicks on article links
    // whose preview text happens to contain words like "allowed" or "continued".
    // \b fails on å/ä/ö, so the Swedish/Nordic matcher uses Unicode-aware
    // lookaround boundaries with the /u flag instead.
    //
    // "allow" and "continue" are deliberately NOT in the generic matcher: login
    // flows (x.com, Google/Apple SSO) render buttons literally labelled
    // "Continue", "Continue with phone", "Allow" — matching them blind fired 4-5
    // phantom, untrusted login-flow submissions per page load and got the account
    // rate-limited before a password was ever entered. The remaining generic
    // words only fire inside a genuine cookie/consent context (see hasConsentCtx).
    const NORDIC_RE = new RegExp(
        '(?<![\\p{L}\\p{N}])(godkänn alla|godkänn|acceptera alla|acceptera|' +
        'tillåt alla|tillåt|jag godkänner|jag förstår|' +
        'godta alle|tillad alle|accepter alle)(?![\\p{L}\\p{N}])', 'iu');
    // A generic accept/agree label is only trusted as a cookie button when the
    // button itself or a nearby ancestor is tagged as consent UI. We check
    // id/class/aria/data attributes up the tree (not ancestor innerText, which
    // for a large wrapper would pick up an unrelated footer "privacy" link).
    const CONSENT_CTX_RE = /(cookie|kakor|consent|samtycke|gdpr|\bcmp\b)/i;
    const hasConsentCtx = (el) => {
        if (/cookie|kakor/i.test(el.innerText || '')) return true;
        let n = el, depth = 0;
        while (n && depth < 6) {
            const cls = typeof n.className === 'string' ? n.className : '';
            const sig = (n.id || '') + ' ' + cls + ' ' +
                (n.getAttribute ? (n.getAttribute('aria-label') || '') + ' ' +
                                  (n.getAttribute('data-testid') || '') : '');
            if (CONSENT_CTX_RE.test(sig)) return true;
            n = n.parentElement; depth++;
        }
        return false;
    };
    const labelMatchers = [
        (el) => {
            const t = (el.innerText||'').trim();
            if (!(t.length > 0 && t.length <= 80)) return false;
            const looksAccept =
                /\b(accept all|accept cookies|i agree|agree|consent|got it)\b/i.test(t) ||
                NORDIC_RE.test(t);
            return looksAccept && hasConsentCtx(el);
        },
        (el) => ['Accept all cookies','Accept cookies','Consent','Close consent Widget',"I'm ok with that"].includes(el.getAttribute('aria-label')||''),
        (el) => ['cookie-consent-button','accept-cookies','acceptAllCookies','onetrust-accept-btn-handler'].includes(el.id),
        (el) => /(?:^|\s)(?:cookie-accept|accept-cookies|cookie-consent)(?:\s|$)/.test(el.className||''),
    ];
    document.querySelectorAll('button, a, [role="button"]').forEach(el => {
        try {
            if (el.offsetParent !== null && labelMatchers.some(m => m(el))) {
                el.click();
                clicked++;
            }
        } catch (e) {}
    });

    // Pass 2 — remove cookie banner containers by ID/class
    const cookieSelectors = [
        '[id*="cookie"]', '[id*="Cookie"]', '[id*="consent"]', '[id*="Consent"]',
        '[class*="cookie"]', '[class*="Cookie"]', '[class*="consent"]', '[class*="Consent"]',
        '[id*="gdpr"]', '[id*="GDPR"]', '[class*="gdpr"]', '[class*="GDPR"]',
        '[id*="banner"]', '[class*="banner"]', '[id*="notice"]', '[class*="notice"]',
        '#onetrust-consent-sdk', '.onetrust-pc-dark-filter', '.optanon-alert-box-wrapper',
        '[class*="privacy"]', '[class*="Privacy"]', '[id*="privacy"]'
    ];
    cookieSelectors.forEach(sel => {
        document.querySelectorAll(sel).forEach(el => {
            const text = (el.innerText || '').toLowerCase();
            if (text.includes('cookie') || text.includes('consent') || text.includes('privacy') ||
                text.includes('gdpr') || text.includes('accept') || text.includes('agree')) {
                safeRemove(el);
            }
        });
    });

    // Pass 3 — kill fixed/sticky overlays parked above content
    const overlaySelectors = [
        '[style*="position: fixed"]', '[style*="position:fixed"]',
        '[style*="position: sticky"]', '[style*="position:sticky"]',
        '[class*="modal"]', '[class*="overlay"]', '[class*="popup"]',
        '[class*="dialog"]', '[class*="backdrop"]', '[role="dialog"]'
    ];
    document.querySelectorAll(overlaySelectors.join(', ')).forEach(el => {
        const style = window.getComputedStyle(el);
        const zIndex = parseInt(style.zIndex) || 0;
        if (zIndex > 1000) {
            const text = (el.innerText || '').toLowerCase();
            if (text.includes('cookie') || text.includes('consent') || text.includes('accept') ||
                text.includes('privacy') || text.includes('gdpr') || el.offsetHeight > 100) {
                safeRemove(el);
            }
        }
    });

    // Re-enable scrolling (cookie banners often disable it)
    if (document.body) {
        document.body.style.overflow = 'auto';
        document.body.style.position = 'static';
    }
    if (document.documentElement) {
        document.documentElement.style.overflow = 'auto';
        document.documentElement.style.position = 'static';
    }
    // Leftover backdrops — whole class names only ('supports-backdrop-filter' is not
    // a backdrop), and only when the element really is an overlay.
    document.querySelectorAll('.backdrop, .overlay, .modal-backdrop').forEach(el => {
        try {
            if (looksLikeOverlay(el)) safeRemove(el);
        } catch (e) {}
    });

    return { clicked: clicked, removed: removed, kept: kept };
})();
"""

    async def _async_remove_cookies_banner(self, tab: 'nodriver.Tab', url: str) -> dict:
        """One consent pass; returns the JS result ({clicked, removed}) or {} on failure."""
        try:
            result = await self._evaluate(tab, self._COOKIE_REMOVAL_JS, timeout=5)
        except asyncio.TimeoutError:
            trace(f"Cookie removal script timed out for {url}, continuing anyway")
            return {}
        except Exception as e:
            trace(f"Cookie removal script failed for {url}: {e}")
            return {}
        return result if isinstance(result, dict) else {}

    @classmethod
    def _is_auth_url(cls, url: str) -> bool:
        """True for login/SSO flows, where the consent pass must be skipped."""
        return bool(url) and cls._AUTH_URL_RE.search(url) is not None

    async def _async_settle_cookie_banner(self, tab: 'nodriver.Tab', url: str):
        """Accept/remove consent UI after page readiness, retrying once when the
        first pass matched nothing (consent SDKs mount asynchronously — the
        banner may not be rendered yet right after readiness)."""
        if self._is_auth_url(url):
            info(f"cookie consent: skipping consent pass on auth/login URL {url} "
                 f"(a stray click there submits the auth flow)")
            return
        result = await self._async_remove_cookies_banner(tab, url)
        if result.get('clicked', 0) or result.get('removed', 0):
            info(f"cookie consent: clicked={result.get('clicked', 0)} "
                 f"removed={result.get('removed', 0)} on {url}")
            self._trace_kept_consent_matches(result, url)
            return
        trace(f"cookie consent: nothing matched on {url} — "
              f"re-checking once in {self._COOKIE_RETRY_DELAY}s (SDK may mount late)")
        await asyncio.sleep(self._COOKIE_RETRY_DELAY)
        result = await self._async_remove_cookies_banner(tab, url)
        if result.get('clicked', 0) or result.get('removed', 0):
            info(f"cookie consent: late-mounting banner handled on retry — "
                 f"clicked={result.get('clicked', 0)} removed={result.get('removed', 0)} on {url}")
        else:
            trace(f"cookie consent: nothing to accept or remove on {url}")
        self._trace_kept_consent_matches(result, url)

    @staticmethod
    def _trace_kept_consent_matches(result: dict, url: str):
        if result.get('kept', 0):
            trace(f"cookie consent: kept {result['kept']} matched element(s) on {url} — "
                  f"<html>/<body> or a wrapper of the page content, not a banner")

    # Query parameters that only track the click: dropped from every link the extractor
    # renders. A page's links are most of its text on a commerce site — booking.com gives
    # each hotel three or four links of 500–900 chars of affiliate and session parameters
    # (prod 2026-11-03: 164k of a 199k-char results page), and the LLM pays to read every
    # byte. Generic ones everywhere, a site's own under its host suffix.
    TRACKING_PARAMS = frozenset({
        'fbclid', 'gclid', 'dclid', 'msclkid', 'yclid', 'twclid', 'ttclid', 'igshid', 'srsltid',
        'mc_cid', 'mc_eid', '_ga', '_gl', '_hsenc', '_hsmi', 'ref_src', 'ref_url', 'spm',
        'vero_id', 'wickedid', 'oly_enc_id', 'oly_anon_id', 'mkt_tok', 'trk', 'trkCampaign',
    })
    TRACKING_PARAM_PREFIXES = ('utm_',)
    SITE_TRACKING_PARAMS = {
        'booking.com': frozenset({
            'aid', 'label', 'sid', 'srpvid', 'srepoch', 'ucfs', 'arphpl', 'all_sr_blocks',
            'highlighted_blocks', 'matching_block_id', 'sr_pri_blocks', 'hpos', 'hapos',
            'sr_order', 'from', 'from_sr', 'dest_id', 'dest_type', 'search_selected',
            'ss', 'ssne', 'ssne_untouched', 'lang', 'soz', 'lang_changed', 'sb_price_type', 'map',
        }),
        'airbnb.com': frozenset({
            'ref_fsid', 'ref_search_session_id', 'federated_search_id', 'search_session_id',
            'source_impression_id', 'previous_page_section_name',
        }),
    }

    @classmethod
    def clean_link(cls, url: str) -> str:
        """The link without its tracking parameters; the rest of the query, the order
        included, stays as it was."""
        parsed = urlparse(url)
        if not parsed.query:
            return url
        host = parsed.netloc.lower()
        site_params = next((params for suffix, params in cls.SITE_TRACKING_PARAMS.items()
                            if host == suffix or host.endswith('.' + suffix)), frozenset())
        kept = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
                if k not in cls.TRACKING_PARAMS and k not in site_params
                and not k.startswith(cls.TRACKING_PARAM_PREFIXES)]
        if len(kept) == parsed.query.count('&') + 1:
            return url
        return urlunparse(parsed._replace(query=urlencode(kept)))

    # A link the LLM has no reason to follow: a wiki's section edit links and its
    # template, file, category and project namespaces. A same-page anchor (a footnote,
    # a table of contents entry) is caught without a pattern, by comparing the pages.
    HOUSEKEEPING_LINK_PATTERNS = (
        re.compile(r'[?&]action=edit\b'),
        re.compile(r'/wiki/(?:Template|Special|Help|Wikipedia|File|Category|Portal|Talk)(?:_talk)?:'),
    )

    @staticmethod
    def _origin(url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else ''

    @classmethod
    def _is_housekeeping_link(cls, href: str, base_page: str) -> bool:
        """True for an absolute link that only points within `base_page` (the page without
        its fragment) or matches HOUSEKEEPING_LINK_PATTERNS."""
        if urlunparse(urlparse(href)._replace(fragment='')) == base_page:
            return True
        return any(p.search(href) for p in cls.HOUSEKEEPING_LINK_PATTERNS)

    @staticmethod
    def render_link(href: str, text: str, base_origin: str) -> str:
        """The link as the LLM reads it: `[text](url)`, the url a root-relative path when
        the link stays on the page's own site — `open_url` and `browse_open` resolve such a
        path against the page it came from."""
        if base_origin and (href == base_origin or href.startswith(base_origin + '/')):
            href = href[len(base_origin):] or '/'
        return f'[{text}]({href})'

    @staticmethod
    def extract_text_with_links(html_content: str, base_url: str, *, visible_only: bool = False) -> str:
        """The readable text of a page. visible_only leaves out what the JSON payloads
        (__NEXT_DATA__, embedded script blocks) add — the test for a page that has
        rendered nothing yet."""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Pull structured data out of Next.js's __NEXT_DATA__ payload before we
        # strip <script> tags. Next.js sites (e.g. Gina Tricot) serve full
        # product/variant/stock data here that's invisible to the visible-text
        # extraction below.
        next_data_text = WebAccess._extract_next_data_summary(soup)
        embedded_json = [script.string for script in soup.find_all('script', type='application/json')
                         if script.string and script.get('id') != '__NEXT_DATA__']

        # Remove script, style, nav, footer, header, and sidebar elements. <template>
        # holds inert markup browsers never render (Reddit ships its cookie dialog
        # that way) but BeautifulSoup would read as page text.
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe',
                             'template']):
            element.decompose()

        # Remove cookie consent containers (Cookiebot, OneTrust, generic banners)
        cookie_id_patterns = ['cookiebot', 'cookiebanner', 'cookie-consent', 'cookie_consent',
                              'cookieconsent', 'onetrust', 'gdpr']
        cookie_elements = soup.find_all(
            id=lambda x: x and any(p in x.lower() for p in cookie_id_patterns)
        )
        for element in cookie_elements:
            element.decompose()

        body = soup.find('body')
        if not body:
            return ""

        # Remove drawer/modal/overlay elements (common SPA navigation patterns).
        # A class match that wraps most of the page is a layout container, not
        # boilerplate — Reddit's page wrapper is "main-container … fixed-sidebar"
        # and holds <main> with the whole thread — so those are kept.
        boilerplate_class_patterns = ['drawer', 'modal', 'overlay', 'sidebar', 'popup',
                                      'offcanvas', 'dropdown-menu']
        body_len = len(body.get_text(strip=True))
        for element in soup.find_all(
            class_=lambda c: c and any(p in ' '.join(c).lower() if isinstance(c, list)
                                       else p in c.lower()
                                       for p in boilerplate_class_patterns)
        ):
            if element.decomposed:
                continue
            if element.find('main') or (
                    body_len and len(element.get_text(strip=True)) / body_len > 0.5):
                trace(f"extract_text: keeping boilerplate-classed layout wrapper "
                      f"<{element.name} class={element.get('class')}> on {base_url}")
                continue
            element.decompose()

        # Try to find main content area
        # Note: We prefer single container elements over 'article' which may have many instances
        main_content = (
            soup.find('main') or
            soup.find(role='main') or
            soup.find('div', {'id': 'content'}) or
            soup.find('div', {'id': 'main'}) or
            soup.find('div', {'class': 'content'})
        )

        # If the candidate holds less than 2/3 of the body's text, significant
        # content lives outside it (e.g. flight tables as siblings of <main>)
        if main_content:
            candidate_len = len(main_content.get_text(strip=True))
            body_len = len(body.get_text(strip=True))
            if body_len > 0 and candidate_len / body_len < 0.67:
                main_content = body
        else:
            main_content = body

        # Absolute URLs without their tracking parameters
        stripped = 0
        for a_tag in main_content.find_all('a', href=True):
            absolute_url = urljoin(base_url, a_tag['href'])
            cleaned = WebAccess.clean_link(absolute_url)
            stripped += len(absolute_url) - len(cleaned)
            a_tag['href'] = cleaned

        # A link rendered once: the second and later <a> to the same URL go in as their
        # text alone (a results page links each item from its image, its name and its
        # price). The LLM has the URL from the first one. A link that leads nowhere the
        # LLM would go (a footnote anchor on the same page, a wiki's edit link) is its
        # text alone too.
        seen_links: set[str] = set()
        repeated = 0
        housekeeping = 0
        base_page = urlunparse(urlparse(base_url)._replace(fragment=''))
        base_origin = WebAccess._origin(base_url)

        # Extract text while keeping the links
        def extract_with_links(element) -> str:
            nonlocal repeated, housekeeping
            result = []
            for child in element.children:
                if isinstance(child, Comment):
                    continue
                if isinstance(child, NavigableString):
                    text = str(child).strip()
                    if text:
                        result.append(text)
                elif child.name == 'a' and child.get('href'):
                    link_text = child.get_text(strip=True)
                    if link_text:  # Only include links with text
                        href = child['href']
                        if href in seen_links:
                            repeated += 1
                            result.append(link_text)
                        elif WebAccess._is_housekeeping_link(href, base_page):
                            housekeeping += 1
                            result.append(link_text)
                        else:
                            seen_links.add(href)
                            result.append(WebAccess.render_link(href, link_text, base_origin))
                elif child.name in ['br']:
                    result.append('\n')
                elif child.name in ['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li',
                                    'table', 'thead', 'tbody', 'tr', 'td', 'th', 'section']:
                    # Recursively process block elements
                    inner = extract_with_links(child)
                    if inner:
                        result.append(inner)
                        result.append('\n')
                else:
                    # Recursively process other elements
                    inner = extract_with_links(child)
                    if inner:
                        result.append(inner)

            return ' '.join(result)

        text = extract_with_links(main_content)
        if stripped or repeated or housekeeping:
            info(f"extract_text: {stripped} chars of tracking parameters stripped from links, "
                 f"{repeated} repeated and {housekeeping} housekeeping link(s) rendered as text "
                 f"on {base_url}")
        else:
            trace(f"extract_text: links clean on {base_url}, nothing stripped")

        # Clean up excessive whitespace
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]
        text = '\n'.join(lines)

        if visible_only:
            return text

        if next_data_text:
            text = f"{text}\n\n{next_data_text}" if text else next_data_text

        # A page that renders client-side ships its content as JSON (Apple's order
        # status lives in <script id="init_data">) — read it when the markup says little.
        if embedded_json and len(text) < WebAccess._EMBEDDED_JSON_THIN_TEXT:
            json_text = WebAccess._summarise_embedded_json(embedded_json)
            if json_text:
                trace(f"extract_text: only {len(text)} chars of visible text on {base_url} — "
                      f"appending {len(json_text)} chars summarised from embedded JSON")
                text = f"{text}\n\n{json_text}" if text else json_text

        return text

    _EMBEDDED_JSON_THIN_TEXT = 600
    _EMBEDDED_JSON_MAX_CHARS = 3000
    _EMBEDDED_JSON_MAX_VALUE = 200

    @staticmethod
    def _summarise_embedded_json(blocks: list) -> str:
        """Flatten the readable leaves (short strings, numbers) of JSON script blocks
        into 'key.path: value' lines, capped at _EMBEDDED_JSON_MAX_CHARS.

        Array items are deduplicated as whole records: a repeated record is dropped, but
        a record keeps every field of its own even when another record has the same value
        (three order items, two of them 'Shipped')."""
        lines, seen_records, size = [], set(), 0

        def walk(node, path, out):
            nonlocal size
            if size >= WebAccess._EMBEDDED_JSON_MAX_CHARS:
                return
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value, path + [str(key)], out)
            elif isinstance(node, list):
                # path carries the parent records' indices (ints, never displayed). A list
                # inside a record only collapses its own siblings, so the record stays
                # whole and compares equal to its twin in another block.
                nested = any(isinstance(part, int) for part in path)
                seen = set() if nested else seen_records
                for index, item in enumerate(node):
                    record = []
                    walk(item, path + [index], record)
                    if not record:
                        continue
                    key = (tuple(path), tuple(record))
                    if key in seen:
                        size -= sum(len(line) + 1 for line in record)
                        continue
                    seen.add(key)
                    out.extend(record)
            elif isinstance(node, bool) or node is None:
                return
            else:
                value = str(node).strip()
                if isinstance(node, str) and (
                        not value or len(value) > WebAccess._EMBEDDED_JSON_MAX_VALUE
                        or value.startswith(('http://', 'https://', '/', '<', '{'))):
                    return
                names = [part for part in path if isinstance(part, str)]
                line = f"{'.'.join(names[-2:])}: {value}" if names else value
                if line not in out:
                    out.append(line)
                    size += len(line) + 1

        for block in blocks:
            try:
                walk(json.loads(block), [], lines)
            except (ValueError, TypeError) as e:
                trace(f"embedded JSON parse failed: {e}")
        if not lines:
            return ""
        return "[Data from embedded JSON]\n" + "\n".join(lines)

    _NEXT_DATA_PRODUCT_PATHS = (
        ('props', 'pageProps', 'product', 'product'),
        ('props', 'pageProps', 'product'),
        ('props', 'pageProps', 'initialProduct'),
        ('props', 'pageProps', 'pdpData', 'product'),
    )

    @staticmethod
    def _extract_next_data_summary(soup: 'BeautifulSoup') -> str:
        script = soup.find('script', id='__NEXT_DATA__')
        if not script or not script.string:
            return ""
        try:
            data = json.loads(script.string)
        except (ValueError, TypeError) as e:
            trace(f"__NEXT_DATA__ JSON parse failed: {e}")
            return ""

        product = None
        for path in WebAccess._NEXT_DATA_PRODUCT_PATHS:
            node = data
            for key in path:
                if not isinstance(node, dict):
                    node = None
                    break
                node = node.get(key)
                if node is None:
                    break
            if isinstance(node, dict) and (node.get('skus') or node.get('variants') or node.get('name')):
                product = node
                break

        if not product:
            return ""

        return WebAccess._format_next_data_product(product)

    @staticmethod
    def _format_next_data_product(product: dict) -> str:
        lines = ["[Structured product data from __NEXT_DATA__]"]

        name = product.get('name')
        code = product.get('code') or product.get('id')
        if name:
            lines.append(f"Product: {name}" + (f" (code {code})" if code else ""))
        elif code:
            lines.append(f"Product code: {code}")

        price = product.get('currentPrice') or product.get('salePrice') or product.get('listPrice')
        old_price = product.get('oldPrice')
        if price is not None:
            price_line = f"Price: {price}"
            if old_price and str(old_price) != str(price):
                price_line += f" (was {old_price})"
            lines.append(price_line)

        overall_stock = product.get('onlineStockStatus') or product.get('stockStatus')
        in_stock = product.get('inStock')
        if overall_stock is not None or in_stock is not None:
            parts = []
            if overall_stock is not None:
                parts.append(str(overall_stock))
            if in_stock is not None:
                parts.append(f"inStock={str(in_stock).lower()}")
            lines.append("Overall stock: " + ", ".join(parts))

        skus = product.get('skus') or product.get('variants') or []
        if isinstance(skus, list) and skus:
            lines.append("Per-size stock:")
            for sku in skus:
                if not isinstance(sku, dict):
                    continue
                size = sku.get('size') or sku.get('name') or sku.get('label') or '?'
                sku_code = sku.get('code') or sku.get('id') or ''
                online = sku.get('onlineStockStatus') or sku.get('stockStatus') or '?'
                in_stock_sku = sku.get('inStock')
                low_stock = sku.get('isLowStock')
                bits = [f"size={size}"]
                if sku_code:
                    bits.append(f"sku={sku_code}")
                bits.append(f"status={online}")
                if in_stock_sku is not None:
                    bits.append(f"inStock={str(in_stock_sku).lower()}")
                if low_stock:
                    bits.append("lowStock=true")
                lines.append("  - " + ", ".join(bits))

        return "\n".join(lines) if len(lines) > 1 else ""

    @staticmethod
    def _is_blocked_response(page_source: str) -> bool:
        if not page_source or len(page_source.strip()) < 50:
            return True
        lower = page_source.lower()
        blocked_markers = [
            'cf-browser-verification',
            'cf_chl_opt',
            'just a moment...</title>',
            'checking your browser',
            'datadome',
            'dd.js',
            'access denied</title>',
            'please enable javascript to view the page content',
            'attention required! | cloudflare',
        ]
        return any(marker in lower for marker in blocked_markers)

    def _fetch_via_cloudscraper(self, url: str) -> str | None:
        try:
            scraper = cloudscraper.create_scraper(
                browser={'browser': 'chrome', 'platform': 'darwin', 'mobile': False}
            )
            response = scraper.get(url, timeout=20)
            if response.status_code == 200 and len(response.text) > 100:
                return response.text
            trace(f"cloudscraper returned status {response.status_code} for {url}")
            return None
        except Exception as e:
            trace(f"cloudscraper failed for {url}: {e}")
            return None

    def _fetch_via_curl_cffi(self, url: str) -> str | None:
        try:
            headers = {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-User": "?1",
                "Sec-Fetch-Dest": "document",
                "Upgrade-Insecure-Requests": "1",
            }
            response = cffi_requests.get(url, impersonate="chrome131", headers=headers, timeout=20)
            if response.status_code == 200 and len(response.text) > 100:
                return response.text
            trace(f"curl_cffi returned status {response.status_code} for {url}")
            return None
        except Exception as e:
            trace(f"curl_cffi failed for {url}: {e}")
            return None

    async def _async_wait_for_dynamic_content(self, tab: 'nodriver.Tab', timeout=5, stable_period=0.5):
        """Wait until the page DOM stops changing."""
        deadline = time.time() + timeout
        previous_len = 0
        stable_since = None

        while time.time() < deadline:
            try:
                content = await tab.get_content()
                current_len = len(content or '')
            except Exception:
                return
            if current_len != previous_len:
                previous_len = current_len
                stable_since = time.time()
            elif stable_since and (time.time() - stable_since) >= stable_period:
                break
            await asyncio.sleep(0.25)

    async def _async_wait_for_page_ready(self, tab: 'nodriver.Tab', ready_timeout=None,
                                         quiet_ms=None, quiet_timeout=None) -> str:
        """Soft readiness gate: wait for readyState, then for the DOM to go quiet.

        WAIT_READY: poll document.readyState until 'interactive'/'complete'.
        WAIT_QUIET: wait until __domMon reports no DOM mutation for quiet_ms
        (falling back to the content-length stability loop when the monitor is
        absent).
        WAIT_NETWORK: then until no captured XHR/fetch is in flight and none was
        sent or answered for _PAGE_NET_IDLE_MS — a quiet DOM may be a shell whose
        data is still on its way (_async_wait_for_network_idle).
        Every cap is SOFT — on expiry or evaluate failure this logs and returns;
        it never raises, never stops the load, never reinits the driver. The
        return value is a status string for logging/tests only.
        """
        ready_timeout = self._PAGE_READY_TIMEOUT if ready_timeout is None else ready_timeout
        quiet_ms = self._PAGE_QUIET_MS if quiet_ms is None else quiet_ms
        quiet_timeout = self._PAGE_QUIET_TIMEOUT if quiet_timeout is None else quiet_timeout

        async def ev(expr):
            return await self._cdp_wait_for(
                self._evaluate(tab, expr), self._PAGE_READY_EVAL_TIMEOUT)

        status = "ready"
        ready_deadline = time.time() + ready_timeout
        while True:
            try:
                state = await ev("document.readyState")
            except Exception as e:
                trace(f"page-ready: readyState evaluate failed "
                      f"({type(e).__name__}: {e}) — proceeding with the page as-is")
                return "soft-timeout:evaluate"
            if state in ('interactive', 'complete'):
                break
            if time.time() >= ready_deadline:
                note(f"page-ready: readyState still {state!r} after {ready_timeout}s — "
                     "proceeding with the page as-is (soft cap)")
                status = "soft-timeout:readyState"
                break
            await asyncio.sleep(self._PAGE_READY_POLL)

        quiet_deadline = time.time() + quiet_timeout
        while time.time() < quiet_deadline:
            try:
                idle_ms = await ev("window.__domMon ? (Date.now() - window.__domMon.last) : -1")
            except Exception as e:
                trace(f"page-ready: DOM-quiet evaluate failed "
                      f"({type(e).__name__}: {e}) — proceeding with the page as-is")
                return "soft-timeout:evaluate"
            if not isinstance(idle_ms, (int, float)) or idle_ms < 0:
                fine("page-ready: __domMon absent — falling back to content-length stability")
                remaining = max(1, int(quiet_deadline - time.time()))
                await self._async_wait_for_dynamic_content(tab, timeout=remaining)
                await self._async_wait_for_network_idle()
                return "fallback:no-dommon"
            if idle_ms >= quiet_ms:
                fine(f"page-ready: DOM quiet for {int(idle_ms)}ms — {status}")
                await self._async_wait_for_network_idle()
                return status
            await asyncio.sleep(self._PAGE_READY_POLL)

        trace(f"page-ready: DOM still mutating after {quiet_timeout}s — "
              "proceeding with the page as-is (soft cap)")
        await self._async_wait_for_network_idle()
        return "soft-timeout:quiet"

    async def _async_wait_for_network_idle(self, idle_ms=None, timeout=None) -> str:
        """WAIT_NETWORK: the third stage of the readiness gate. Returns once no captured
        request is in flight and none was sent or answered for idle_ms; a soft cap of
        timeout seconds, like the other stages. Pages whose capture recorded nothing
        (handlers not installed, a fetch-free page) pass straight through."""
        idle_ms = self._PAGE_NET_IDLE_MS if idle_ms is None else idle_ms
        timeout = self._PAGE_NET_IDLE_TIMEOUT if timeout is None else timeout
        last = getattr(self, '_network_last_activity', 0.0)
        if not last:
            return "idle:no-traffic"
        started = time.time()
        deadline = started + timeout
        while True:
            pending = self._network_in_flight()
            idle_for = (time.time() - self._network_last_activity) * 1000
            if not pending and idle_for >= idle_ms:
                if time.time() - started >= self._PAGE_READY_POLL:
                    fine(f"page-ready: network idle after {time.time() - started:.1f}s")
                return "idle"
            if time.time() >= deadline:
                busy = (', '.join(f"{e['method']} {e['url']}" for e in pending[:3])
                        or 'nothing in flight, traffic still arriving')
                trace(f"page-ready: network still busy after {timeout}s ({busy}) — "
                      "proceeding with the page as-is (soft cap)")
                return "soft-timeout:network"
            await asyncio.sleep(self._PAGE_READY_POLL)

    @staticmethod
    def _bidi_to_python(node):
        """Convert a CDP DeepSerializedValue tree into native Python values.

        nodriver requests deep serialization with no way to opt out, so even simple
        results come back as a BiDi-formatted tree. The top-level node arrives as
        a `DeepSerializedValue` dataclass; nested entries arrive as plain dicts
        with `type` / `value` keys. Handle both shapes.
        """
        if node is None:
            return None
        if isinstance(node, dict):
            type_ = node.get('type')
            value = node.get('value')
        else:
            type_ = getattr(node, 'type_', None)
            value = getattr(node, 'value', None)
        if type_ in ('null', 'undefined'):
            return None
        if type_ in ('number', 'string', 'boolean', 'bigint'):
            return value
        if type_ == 'array':
            if not value:
                return []
            return [WebAccess._bidi_to_python(item) for item in value]
        if type_ == 'object':
            if not value:
                return {}
            out = {}
            for entry in value:
                if isinstance(entry, (list, tuple)) and len(entry) == 2:
                    key, item = entry
                    out[key] = WebAccess._bidi_to_python(item)
            return out
        return value

    @staticmethod
    async def _evaluate(tab: 'nodriver.Tab', expression: str, *, return_by_value: bool = True,
                        timeout: Optional[float] = None):
        """Wrapper around tab.evaluate that normalizes nodriver's RemoteObject return quirk.

        With deep serialization (which nodriver always uses), the JS result lands in
        `deep_serialized_value` rather than `value`, so nodriver's own `return_by_value`
        check misses it and returns the raw RemoteObject. We unwrap it here.

        Every evaluate is bounded (default _EVAL_TIMEOUT): a target mid-renderer-swap
        never answers, and an unbounded await here is what wedged checkout clicks.
        Expiry raises asyncio.TimeoutError; the late reply is consumed safely
        (_cdp_wait_for shields).
        """
        timeout = WebAccess._EVAL_TIMEOUT if timeout is None else timeout
        result = await WebAccess._cdp_wait_for(
            tab.evaluate(expression, return_by_value=return_by_value, await_promise=True),
            timeout)
        try:
            from nodriver.cdp import runtime as cdp_runtime
        except Exception:
            return result
        if isinstance(result, cdp_runtime.RemoteObject):
            if getattr(result, 'value', None) is not None:
                return result.value
            deep = getattr(result, 'deep_serialized_value', None)
            if deep is not None:
                return WebAccess._bidi_to_python(deep)
            return None
        return result

    # ------------------------------------------------------------------
    # Cross-origin iframe (OOPIF) access.
    #
    # With site isolation, cross-origin iframes run out-of-process: they are
    # invisible to parent-page JS (same-origin policy) AND absent from the main
    # target's DOM tree — Chrome exposes them as separate CDP targets of type
    # "iframe". Those targets serve no per-target websocket endpoint (404), so
    # the only way in is Target.attachToTarget(flatten=true) on an existing
    # connection, then session-scoped commands. Evaluating there runs inside
    # the frame's own origin, where its document is directly scriptable.

    async def _send_on_session(self, cdp_obj, session_id: str):
        """Send a CDP command scoped to a flat-protocol session over the main tab's connection.

        The reply await is bounded: a frame target destroyed by navigation simply never
        answers, and waiting on it forever is what wedged whole browse commands. Expiry
        raises asyncio.TimeoutError (shielded, so a late reply is consumed harmlessly)."""
        conn = self._tab
        if conn.closed:
            await conn.connect()
        tx = Transaction(cdp_obj)
        tx.connection = conn
        tx.id = next(conn.__count__)
        conn.mapper[tx.id] = tx
        await conn.websocket.send(json.dumps({
            "method": tx.method, "params": tx.params,
            "id": tx.id, "sessionId": str(session_id),
        }))
        return await self._cdp_wait_for(tx, self._FRAME_SEND_TIMEOUT)

    async def _async_attach_frame_session(self, target_id) -> str:
        """Return a devtools session attached to the given iframe target, creating one if needed."""
        key = str(target_id)
        cached = self._browse_frame_sessions.get(key)
        if cached:
            return cached
        session_id = str(await self._cdp_wait_for(
            self._tab.send(
                cdp.target.attach_to_target(target_id=cdp.target.TargetID(key), flatten=True)),
            self._FRAME_ATTACH_TIMEOUT,
        ))
        self._browse_frame_sessions[key] = session_id
        fine(f"Attached devtools session {session_id[:8]}… to iframe target {key[:8]}…")
        return session_id

    async def _evaluate_in_frame(self, session_id: str, expression: str):
        """Runtime.evaluate inside an attached iframe target's session; raises on JS exceptions."""
        remote_object, errors = await self._send_on_session(
            cdp.runtime.evaluate(
                expression=expression,
                return_by_value=True,
                await_promise=True,
                user_gesture=True,
                allow_unsafe_eval_blocked_by_csp=True,
            ),
            session_id,
        )
        if errors:
            exc = getattr(errors, 'exception', None)
            desc = getattr(exc, 'description', None) or getattr(errors, 'text', None) or str(errors)
            raise RuntimeError(f"frame JS threw: {desc}")
        return getattr(remote_object, 'value', None) if remote_object else None

    async def _async_eval_in_frame_target(self, target_id, expression: str):
        """Evaluate in an iframe target, re-attaching once if the cached session is stale.

        Only ProtocolException triggers the retry (dead session/target — the command
        never ran, so re-sending is safe even for non-idempotent JS like clicks);
        JS exceptions propagate immediately.
        """
        key = str(target_id)
        for attempt in (0, 1):
            session_id = await self._async_attach_frame_session(target_id)
            try:
                return await self._evaluate_in_frame(session_id, expression)
            except ProtocolException:
                self._browse_frame_sessions.pop(key, None)
                if attempt:
                    raise

    # Supported keys for real CDP key dispatch: lowercase name -> (key, code,
    # windows_virtual_key_code, text). Keys that produce a character (Enter/Tab)
    # carry `text` on keyDown so CDP emits the char event frameworks listen for.
    _KEY_DEFS: ClassVar[dict] = {
        'enter': ('Enter', 'Enter', 13, '\r'),
        'escape': ('Escape', 'Escape', 27, ''),
        'tab': ('Tab', 'Tab', 9, '\t'),
        'backspace': ('Backspace', 'Backspace', 8, ''),
        'arrowup': ('ArrowUp', 'ArrowUp', 38, ''),
        'arrowdown': ('ArrowDown', 'ArrowDown', 40, ''),
        'arrowleft': ('ArrowLeft', 'ArrowLeft', 37, ''),
        'arrowright': ('ArrowRight', 'ArrowRight', 39, ''),
    }
    _KEY_ALIASES: ClassVar[dict] = {
        'esc': 'escape', 'return': 'enter',
        'up': 'arrowup', 'down': 'arrowdown', 'left': 'arrowleft', 'right': 'arrowright',
    }

    @classmethod
    def _resolve_key(cls, key_name: str) -> Optional[tuple]:
        name = (key_name or '').strip().lower()
        return cls._KEY_DEFS.get(cls._KEY_ALIASES.get(name, name))

    async def _async_dispatch_key(self, key_name: str, frame_id=None) -> None:
        """Press one real key via CDP Input.dispatchKeyEvent (trusted events).

        keyDown with `text` makes CDP synthesize rawKeyDown+char (the Puppeteer
        approach), so React/SPA handlers see it as genuine typing. With frame_id
        set, the OOPIF's flat session is tried first — Input routing on OOPIF
        sessions is best-effort, so a ProtocolException falls back to main-tab
        dispatch (browser-level key routing follows focus).
        """
        defs = self._resolve_key(key_name)
        if defs is None:
            raise ValueError(f"unsupported key: {key_name!r}")
        key, code, vk, text = defs
        down_kwargs = dict(type_='keyDown', key=key, code=code,
                           windows_virtual_key_code=vk, native_virtual_key_code=vk)
        if text:
            down_kwargs.update(text=text, unmodified_text=text)
        up_kwargs = dict(type_='keyUp', key=key, code=code,
                         windows_virtual_key_code=vk, native_virtual_key_code=vk)

        if frame_id is not None:
            try:
                session_id = await self._async_attach_frame_session(frame_id)
                await self._send_on_session(cdp.input_.dispatch_key_event(**down_kwargs), session_id)
                await self._send_on_session(cdp.input_.dispatch_key_event(**up_kwargs), session_id)
                trace(f"dispatched {key} via iframe session {str(session_id)[:8]}…")
                return
            except ProtocolException as e:
                note(f"key dispatch on iframe session failed ({e}) — falling back to main-tab dispatch")
        await self._cdp_wait_for(
            self._tab.send(cdp.input_.dispatch_key_event(**down_kwargs)), self._EVAL_TIMEOUT)
        await self._cdp_wait_for(
            self._tab.send(cdp.input_.dispatch_key_event(**up_kwargs)), self._EVAL_TIMEOUT)

    async def _async_live_target_ids(self) -> Optional[set]:
        """The ids of the targets Chrome still has (Target.getTargets), or None when the call
        fails. nodriver drops a target only when Chrome announces its destruction, and an
        iframe target's can go unannounced, so without this a tracking pixel's frame from an
        earlier page is attached to, and timed out on, at every snapshot of the session."""
        try:
            infos = await self._cdp_wait_for(
                self._browser.connection.send(cdp.target.get_targets()), self._EVAL_TIMEOUT)
        except Exception as e:
            trace(f"frame targets: Target.getTargets failed — keeping nodriver's list: {e}")
            return None
        return {str(ti.target_id) for ti in infos or []}

    async def _async_list_frame_targets(self) -> list:
        """Return TargetInfo for each out-of-process iframe target worth snapshotting."""
        if self._browser is None:
            return []
        try:
            await self._browser.update_targets()
        except Exception as e:
            trace(f"update_targets failed: {e}")
            return []
        live = await self._async_live_target_ids()
        closed = 0
        # An iframe target whose URL is the page itself is the previous page's
        # frame lingering after browse_open() navigated top-level to that same URL
        # (the "open the iframe directly" pattern): its session is dead and every
        # command to it burns the full _FRAME_SEND_TIMEOUT.
        own_label = self._frame_label(self._current_tab_url())
        frames = []
        blank_ids = set()
        for t in self._browser.targets:
            ti = getattr(t, 'target', None)
            if ti is None or ti.type_ != 'iframe':
                continue
            if live is not None and str(ti.target_id) not in live:
                closed += 1
                continue
            url = ti.url or ''
            if not url or url == 'about:blank':
                # A freshly created OOPIF that has not committed its URL yet —
                # nothing to collect, but the snapshot's settle wait must know
                # it is coming rather than silently miss it.
                blank_ids.add(str(ti.target_id))
                continue
            if any(p in url for p in self._SKIP_IFRAME_PATTERNS):
                continue
            if own_label and self._frame_label(url) == own_label:
                trace(f"frame targets: skipping stale iframe target {own_label} — it is the page itself")
                continue
            frames.append(ti)
        self._blank_frame_target_ids = frozenset(blank_ids)
        if closed:
            trace(f"frame targets: skipped {closed} closed iframe target(s) nodriver still lists")
        if blank_ids:
            trace(f"frame targets: {len(blank_ids)} iframe target(s) without a URL yet")
        # Prune sessions for targets that no longer exist (frame navigated away/removed)
        live_ids = {str(ti.target_id) for ti in frames}
        for key in list(self._browse_frame_sessions):
            if key not in live_ids:
                del self._browse_frame_sessions[key]
        return frames

    async def _async_settled_frame_targets(self, cross_origin_srcs: list) -> list:
        """List OOPIF targets, waiting briefly for the ones the parent DOM already
        shows but Chrome has not committed yet.

        The post-action snapshot fires as soon as the parent DOM is quiet; an
        iframe inserted at the end of that burst (Stripe's card field, the
        hCaptcha challenge) becomes a CDP target ~0.5s later, first with an
        empty URL. Reconcile the DOM's cross-origin iframe srcs (by host+path)
        against the target list and re-poll until every src has a target and no
        target is still blank, up to _OOPIF_SETTLE_TIMEOUT. Whatever is still
        unresolved at the deadline is remembered and never waited for again in
        this session (an iframe src that redirects can never match).
        """
        wanted = {self._frame_label(self._normalize_url(src)) for src in cross_origin_srcs if src}
        wanted -= self._oopif_settle_giveups
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._OOPIF_SETTLE_TIMEOUT
        while True:
            frames = await self._async_list_frame_targets()
            have = {self._frame_label(ti.url) for ti in frames}
            missing = wanted - have
            blank = set(self._blank_frame_target_ids) - self._oopif_settle_giveups
            if not missing and not blank:
                return frames
            if loop.time() >= deadline:
                note(f"Snapshot: OOPIF target(s) still not committed after "
                     f"{self._OOPIF_SETTLE_TIMEOUT:.1f}s — not in the target list: "
                     f"{sorted(missing) or 'none'}; without a URL: {len(blank)} — "
                     "collecting without them (not waited for again this session)")
                self._oopif_settle_giveups = self._oopif_settle_giveups | missing | blank
                return frames
            trace(f"Snapshot: waiting for OOPIF target(s) — missing {sorted(missing) or 'none'}, "
                  f"{len(blank)} without a URL yet")
            await asyncio.sleep(self._OOPIF_SETTLE_POLL)

    async def _async_hidden_frames(self, main_result: dict, frame_targets: list,
                                   deadline: float) -> tuple[set, set]:
        """The iframe targets whose hosting <iframe> is collapsed or hidden in the
        document that embeds it — the main page's own, plus what each iframe target
        reports about its children (one cheap eval per target, within the snapshot's
        frame budget). Returns (exact URLs, labels), each minus whatever some document
        shows. The exact URL tells Stripe's frames apart: it mounts several under one
        path, differing only in the fragment, and the express-pay bar that renders
        must not shelter the 8px autofill frame that lists card fields. The label
        fallback covers a src that does not match its target's URL (a redirect).
        """
        def src_of(entry) -> str:
            return entry.get('src', '') if isinstance(entry, dict) else (entry or '')

        def urls(entries) -> set:
            return {self._normalize_url(src_of(e)) for e in (entries or []) if src_of(e)}

        def labels(entries) -> set:
            return {self._frame_label(u) for u in urls(entries)}

        def sized(entries) -> list[str]:
            return [f"{self._frame_label(src_of(e))} {e.get('w')}x{e.get('h')}"
                    + (f" {e.get('display')}" if e.get('display') not in (None, 'block') else '')
                    + (f" {e.get('visibility')}" if e.get('visibility') not in (None, 'visible') else '')
                    + (f" ({e.get('why')})" if e.get('why') else '')
                    if isinstance(e, dict) else self._frame_label(src_of(e))
                    for e in (entries or []) if src_of(e)]

        shown_urls = urls(main_result.get('crossOriginFrames'))
        hidden_urls = urls(main_result.get('hiddenFrames'))
        if hidden_urls:
            trace(f"Snapshot: the page hides iframe(s): {sized(main_result.get('hiddenFrames'))}")
        if len(frame_targets) > 1:
            loop = asyncio.get_running_loop()
            for ti in frame_targets:
                if loop.time() > deadline:
                    trace("Snapshot: frame budget spent before the hidden-iframe pre-pass finished")
                    break
                try:
                    children = await self._async_eval_in_frame_target(
                        ti.target_id, self._FRAME_CHILD_IFRAMES_JS)
                except Exception as e:
                    trace(f"Snapshot: child-iframe pre-pass failed in {self._frame_label(ti.url)}: "
                          f"{type(e).__name__}: {e}")
                    continue
                if isinstance(children, dict):
                    if children.get('shown') or children.get('hidden'):
                        trace(f"Snapshot: {self._frame_label(ti.url)} hosts — shown: "
                              f"{sized(children.get('shown'))}; hidden: {sized(children.get('hidden'))}")
                    shown_urls |= urls(children.get('shown'))
                    hidden_urls |= urls(children.get('hidden'))
        hidden_exact = hidden_urls - shown_urls
        hidden_labels = {self._frame_label(u) for u in hidden_urls} - {self._frame_label(u) for u in shown_urls}
        if hidden_exact or hidden_labels:
            trace(f"Snapshot: hidden iframe(s): {sorted(self._frame_label(u) for u in hidden_exact)}"
                  + (f"; by label only: {sorted(hidden_labels)}" if hidden_labels - {self._frame_label(u) for u in hidden_exact} else ''))
        return hidden_exact, hidden_labels

    @staticmethod
    def _frame_label(url: str) -> str:
        """Short human-readable frame identifier: host + path, no query."""
        try:
            p = urlparse(url)
            label = (p.netloc or '') + (p.path or '')
            return label.rstrip('/') or url
        except Exception:
            return url

    # ------------------------------------------------------------------
    # Popup / new-tab target tracking.
    #
    # Pages (notably payment flows: Klarna Direkt, BankID, OAuth) open the next
    # step via window.open(), which lands in a brand-new page target the browse
    # session would otherwise never see. nodriver's browser connection already
    # discovers those targets (Target.targetCreated + update_targets polling) —
    # including popups spawned from inside cross-origin iframe sessions, since
    # target discovery is browser-global. We keep a stack: a newly created page
    # target becomes the active tab for snapshot/act/inspect/eval/extract/
    # screenshot, and when it closes we fall back to its parent. Every switch
    # queues a note that the next snapshot surfaces so the LLM knows what it is
    # looking at.

    def _page_targets(self) -> list:
        """Current page-type targets from nodriver's registry."""
        if self._browser is None:
            return []
        pages = []
        for t in self._browser.targets:
            ti = getattr(t, 'target', None)
            if ti is not None and ti.type_ == 'page':
                pages.append(t)
        return pages

    @staticmethod
    def _is_popup_candidate_url(url: str) -> bool:
        """True unless the target is browser chrome (devtools, extensions, settings)."""
        return not (url or '').startswith(('devtools://', 'chrome-extension://', 'chrome://'))

    def _as_tab(self, t) -> 'nodriver.Tab':
        """Promote a bare Connection (from update_targets polling) to a full Tab."""
        if hasattr(t, 'evaluate'):
            return t
        return nodriver.Tab(t.websocket_url, target=t.target, browser=self._browser)

    @staticmethod
    def _tab_id(tab) -> str:
        return str(tab.target.target_id)

    def _registry_target(self, tid: str):
        """nodriver registry object for a target id, or None if not yet discovered."""
        for t in self._page_targets():
            if str(t.target.target_id) == tid:
                return t
        return None

    async def _async_live_page_targets(self) -> Optional[dict]:
        """Authoritative live page targets via CDP Target.getTargets.

        Returns {target_id: TargetInfo} for page-type targets, or None when the
        transport is unreachable — in that case callers must mutate no target
        bookkeeping and leave recovery to the dead-browser reinit path. Also
        refreshes nodriver's registry (needed by _as_tab) best-effort."""
        if self._browser is None:
            return None
        try:
            infos = await self._browser.connection.send(cdp.target.get_targets())
        except Exception as e:
            trace(f"popup: Target.getTargets failed — live targets unknown: {e}")
            return None
        try:
            await self._browser.update_targets()
        except Exception as e:
            trace(f"popup: update_targets after getTargets failed: {e}")
        return {str(ti.target_id): ti for ti in infos or [] if ti.type_ == 'page'}

    def _adopt_newest_live_target(self, live: dict) -> bool:
        """Make the newest verified-live page target the new base tab.

        Used when the base tab (and any popup parents) died. Returns False when
        no adoptable live target exists — then the tab is cleared so the next
        browse command goes through the browser reinit path instead of a ghost."""
        self._popup_parents = []
        self._browse_frame_sessions = {}
        for tid in reversed(list(live)):
            url = getattr(live[tid], 'url', '') or ''
            if not self._is_popup_candidate_url(url):
                continue
            t = self._registry_target(tid)
            if t is None:
                trace(f"popup: live target {tid[:8]}… not in nodriver registry — skipping")
                continue
            self._tab = self._as_tab(t)
            info(f"popup: base tab is dead — adopting live target {tid[:8]}… ({url}) as the new base")
            self._pending_target_notes.append(
                f"NOTE: the previous tab is gone — the active page is now {url}. "
                "Element indices from earlier snapshots are invalid."
            )
            return True
        info("popup: no live page target left to adopt — deferring to the browser reinit path")
        self._tab = None
        return False

    async def _async_wait_for_fresh_target_url(self, tid: str) -> Optional[str]:
        """Inline grace for window.open popups: poll for a fresh about:blank
        target to navigate (payment/BankID popups open blank, then navigate
        within milliseconds). Returns the last seen URL ('' if still blank), or
        None if the target vanished or the transport failed."""
        deadline = time.time() + self._POPUP_BLANK_GRACE
        url = ''
        while time.time() < deadline:
            await asyncio.sleep(self._POPUP_BLANK_POLL)
            live = await self._async_live_page_targets()
            if live is None or tid not in live:
                return None
            url = getattr(live[tid], 'url', '') or ''
            if url and url != 'about:blank':
                break
        return url

    async def _async_adopt_popup_target(self, tid: str, url: str):
        """Push the current tab and make a verified-live popup target the active tab."""
        t = self._registry_target(tid)
        if t is None:
            trace(f"popup: target {tid[:8]}… not in nodriver registry yet — deferring adoption")
            self._pending_popup_targets.setdefault(tid, time.time())
            return
        tab = self._as_tab(t)
        self._popup_parents.append(self._tab)
        self._tab = tab
        self._browse_frame_sessions = {}  # sessions were scoped to the old tab's connection
        try:
            await tab.send(cdp.page.enable())
        except Exception as e:
            trace(f"popup: page.enable() failed for {tid[:8]}…: {e}")
        self._install_dialog_handler(tab)
        self._install_network_capture(tab)
        info(f"popup: new window/tab target {tid[:8]}… ({url}) — making it the active browse target")
        await self._async_wait_for_page_ready(tab)
        self._pending_target_notes.append(
            f"NOTE: the page opened a popup/new window ({url}). "
            "The snapshot below shows the POPUP, which is now the active page for "
            "browse_act/inspect/eval/extract/screenshot. When it closes you will be "
            "switched back to the parent page automatically."
        )

    async def _async_sync_popup_targets(self):
        """Detect popup creation/destruction and retarget the browse session.

        Called after actions and before snapshots. Falls back through the parent
        stack when the active popup (or a parent under it) has closed, then
        adopts newly created page targets — but only targets verified live with
        a real URL; fresh about:blank targets wait in the pending set until they
        navigate. On transport failure nothing is mutated: the dead-browser
        reinit path owns recovery.
        """
        if self._browser is None or self._tab is None:
            return
        live = await self._async_live_page_targets()
        if live is None:
            trace("popup sync: live-target fetch failed — leaving target bookkeeping untouched")
            return

        while self._popup_parents and self._tab_id(self._tab) not in live:
            closed_id = self._tab_id(self._tab)
            self._tab = self._popup_parents.pop()
            self._browse_frame_sessions = {}
            url = getattr(self._tab.target, 'url', '') or ''
            info(f"popup sync: active popup {closed_id[:8]}… closed — back to {url}")
            self._pending_target_notes.append(
                f"NOTE: the popup window closed — you are now back on the parent page ({url})."
            )

        if self._tab_id(self._tab) not in live:
            info(f"popup sync: active tab {self._tab_id(self._tab)[:8]}… is dead with no live parent")
            if not self._adopt_newest_live_target(live):
                self._known_page_targets = set(live)
                self._pending_popup_targets = {}
                return

        # Revisit pending blank targets before the fresh scan: adopt any that
        # navigated, drop any that closed or stayed blank past the expiry.
        now = time.time()
        for tid, first_seen in list(self._pending_popup_targets.items()):
            if tid not in live:
                trace(f"popup sync: pending blank target {tid[:8]}… closed before navigating — dropping")
                del self._pending_popup_targets[tid]
                continue
            url = getattr(live[tid], 'url', '') or ''
            if url and url != 'about:blank':
                del self._pending_popup_targets[tid]
                if self._is_popup_candidate_url(url):
                    await self._async_adopt_popup_target(tid, url)
                else:
                    trace(f"popup sync: pending target {tid[:8]}… navigated to non-page-content {url} — ignoring")
            elif now - first_seen > self._POPUP_PENDING_EXPIRY:
                trace(f"popup sync: pending target {tid[:8]}… still blank after "
                      f"{self._POPUP_PENDING_EXPIRY:.0f}s — forgetting it")
                del self._pending_popup_targets[tid]

        fresh = [tid for tid in live if tid not in self._known_page_targets]
        self._known_page_targets = set(live)
        active_ids = {self._tab_id(self._tab)} | {self._tab_id(t) for t in self._popup_parents}
        for tid in fresh:
            if tid in active_ids or tid in self._pending_popup_targets:
                trace(f"popup sync: fresh target {tid[:8]}… already tracked — not adopting")
                continue
            url = getattr(live[tid], 'url', '') or ''
            if not self._is_popup_candidate_url(url):
                trace(f"popup sync: ignoring new non-page-content target {tid[:8]}… ({url})")
                continue
            if not url or url == 'about:blank':
                url = await self._async_wait_for_fresh_target_url(tid)
                if url is None:
                    trace(f"popup sync: fresh blank target {tid[:8]}… vanished during grace poll")
                    continue
                if not url or url == 'about:blank':
                    trace(f"popup sync: fresh target {tid[:8]}… still on about:blank — deferring adoption")
                    self._pending_popup_targets[tid] = time.time()
                    continue
                if not self._is_popup_candidate_url(url):
                    trace(f"popup sync: fresh target {tid[:8]}… navigated to non-page-content {url} — ignoring")
                    continue
            await self._async_adopt_popup_target(tid, url)

    async def _async_close_popup_targets(self):
        """Close leftover popup tabs, restore a verified-live base tab, and re-seed known targets."""
        self._pending_target_notes = []
        self._pending_popup_targets = {}
        if self._browser is None:
            self._popup_parents = []
            self._known_page_targets = set()
            return
        live = await self._async_live_page_targets()
        if live is None:
            # Transport is down: mutate no tab bookkeeping — the dead-browser
            # reinit path rebuilds everything via _async_initialize_driver.
            info("popup: live-target fetch failed during cleanup — deferring to the browser reinit path")
            return
        if self._popup_parents:
            base = self._popup_parents[0]
            base_tid = self._tab_id(base)
            leftovers = [t for t in self._popup_parents[1:] + [self._tab]
                         if t is not None and self._tab_id(t) != base_tid]
            live_leftovers = [t for t in leftovers if self._tab_id(t) in live]
            if len(live_leftovers) < len(leftovers):
                trace(f"popup: {len(leftovers) - len(live_leftovers)} leftover popup target(s) already gone")
            if live_leftovers:
                info(f"popup: closing {len(live_leftovers)} leftover popup tab(s)")
            for t in live_leftovers:
                try:
                    await self._browser.connection.send(
                        cdp.target.close_target(target_id=t.target.target_id)
                    )
                    live.pop(self._tab_id(t), None)
                except Exception as e:
                    trace(f"popup: closing leftover popup failed: {e}")
            self._popup_parents = []
            self._browse_frame_sessions = {}
            if base_tid in live:
                info("popup: restored base tab")
                self._tab = base
            else:
                info(f"popup: base tab {base_tid[:8]}… is dead")
                self._adopt_newest_live_target(live)
        elif self._tab is not None and self._tab_id(self._tab) not in live:
            info(f"popup: active tab {self._tab_id(self._tab)[:8]}… is dead")
            self._adopt_newest_live_target(live)
        self._known_page_targets = set(live)

    def _consume_target_notes(self) -> str:
        """Drain queued target-switch notes into a snapshot header prefix."""
        if not self._pending_target_notes:
            return ''
        notes = '\n'.join(self._pending_target_notes)
        self._pending_target_notes = []
        return notes + '\n\n'

    async def _async_safe_stop_page_load(self, tab: 'nodriver.Tab'):
        """Try to stop page load with a short timeout; reinitialize browser if renderer is hung.

        No longer referenced from browse_open (its navigation cap is soft now);
        only the fetch_webpage nodriver tier still uses this on load timeout."""
        try:
            await asyncio.wait_for(tab.evaluate("window.stop();"), timeout=3)
        except Exception as e:
            note(f"window.stop() failed (renderer likely hung): {e} — reinitializing driver")
            await self._async_initialize_driver()

    # The tab strips of a page: a row of 3–12 sibling buttons with one marked active. Only the
    # labels are read — never the panels. Clicking through them cost 1.1 s a tab and, on a shop
    # page whose "tabs" were accessory categories, returned 300k chars of product grids (one
    # 166k-token cache write, 2026-10-18). The open tab's panel is in the DOM already; the rest is
    # one browser-toolkit detour away, and the note at the end of the page says so.
    _TAB_LABELS_JS = r"""
(function() {
    const MAX_TABS = 12;
    const activeBtns = Array.from(document.querySelectorAll('button[class*="active"]'));
    let bestGroup = null;
    for (const btn of activeBtns) {
        const parent = btn.parentElement;
        if (!parent) continue;
        const tabs = Array.from(parent.children).filter(c => c.tagName === 'BUTTON');
        if (tabs.length < 3 || tabs.length > MAX_TABS) continue;
        if (!bestGroup || tabs.length > bestGroup.length) bestGroup = tabs;
    }
    if (!bestGroup) return null;
    return bestGroup.map(b => (b.innerText || '').trim());
})();
"""

    async def _async_tab_labels(self, tab: 'nodriver.Tab') -> list[str]:
        """The labels of the page's tab strip, [] when it has none. No clicks, one evaluate."""
        try:
            labels = await self._evaluate(tab, self._TAB_LABELS_JS, timeout=5)
        except Exception as e:
            trace(f"Tab strip detection failed: {e}")
            return []
        return [str(l) for l in labels] if isinstance(labels, list) else []

    @staticmethod
    def tab_note(labels: list[str]) -> str:
        """The line appended to a fetched page naming the tabs whose panels were not read, or ''
        when fewer than two readable labels remain (icon-only buttons carry no text)."""
        seen, names = set(), []
        for label in labels:
            name = ' '.join(str(label).split())
            if len(name) > 60 or not any(ch.isalnum() for ch in name) or name in seen:
                continue
            seen.add(name); names.append(name)
        if len(names) < 2:
            return ''
        return (f"[This page has tabs whose contents are not shown here: {', '.join(names)}. "
                f"The `browser` toolkit can open the page and click one "
                f"(browse_open, browse_act, browse_extract).]")

    @staticmethod
    def _eval_number(result) -> Optional[float]:
        """A numeric evaluate result, else None — nodriver hands back an ExceptionDetails
        object (not an exception) when the expression throws in the page."""
        if isinstance(result, (int, float)) and not isinstance(result, bool):
            return result
        return None

    async def _async_scroll_to_load_content(self, tab: 'nodriver.Tab'):
        """Scroll through the page to trigger lazy-loaded content."""
        try:
            page_height = self._eval_number(await self._evaluate(tab, 'document.body.scrollHeight'))
            if page_height is None:
                trace("Scroll-to-load skipped: document.body.scrollHeight did not evaluate "
                      "to a number (no <body>?)")
                return
            viewport_height = self._eval_number(await self._evaluate(tab, 'window.innerHeight')) or 800
            position = 0
            max_scrolls = 15
            scrolls = 0
            while position < page_height and scrolls < max_scrolls:
                position += viewport_height
                await tab.evaluate(f'window.scrollTo(0, {position});')
                await asyncio.sleep(0.3)
                new_height = self._eval_number(
                    await self._evaluate(tab, 'document.body.scrollHeight')) or 0
                if new_height > page_height:
                    page_height = new_height
                scrolls += 1
            await tab.evaluate('window.scrollTo(0, 0);')
        except Exception as e:
            trace(f"Scroll-to-load failed: {e}")

    _SKIP_IFRAME_PATTERNS = ('maps.google.com', 'google.com/maps', 'cookiebot.com')

    async def _async_extract_iframe_texts(self, tab: 'nodriver.Tab', base_url: str) -> str:
        """Extract rendered text from each visible iframe and concatenate."""
        iframe_texts = []
        try:
            # query_selector_all, not select_all: select_all polls for up to 10s
            # waiting for a match, turning every extract on an iframe-less page
            # into a fixed 10-second stall.
            iframes = await tab.query_selector_all('iframe')
        except Exception as e:
            trace(f"iframe enumeration failed: {e}")
            return ""

        for i, iframe in enumerate(iframes or []):
            src = ''
            try:
                src = (iframe.attrs.get('src') if iframe.attrs else '') or ''
            except Exception:
                pass
            if any(p in src for p in self._SKIP_IFRAME_PATTERNS):
                continue
            url = src or base_url
            if url.startswith('//'):
                url = 'https:' + url
            elif not url.startswith(('http://', 'https://')):
                url = urljoin(base_url, url)
            try:
                # Read the iframe's body outerHTML via JS that uses contentDocument.
                # Cross-origin iframes will throw inside the try/catch and return ''.
                inner_html = await self._evaluate(
                    tab,
                    f"""
                    (function() {{
                        const f = document.querySelectorAll('iframe')[{i}];
                        if (!f) return '';
                        try {{
                            const doc = f.contentDocument;
                            if (!doc || !doc.body) return '';
                            return doc.body.outerHTML;
                        }} catch (e) {{ return ''; }}
                    }})();
                    """,
                )
            except Exception as e:
                trace(f"Failed to extract iframe {i}: {e}")
                continue
            if inner_html:
                text = self.extract_text_with_links(inner_html, url)
                if text and len(text.strip()) > 50:
                    fine(f"Extracted {len(text)} chars from iframe {i} ({url[:80]})")
                    iframe_texts.append(text)

        return "\n\n".join(iframe_texts)

    async def _async_extract_frame_target_texts(self) -> str:
        """Extract rendered text from cross-origin iframe targets via per-frame sessions."""
        texts = []
        for ti in await self._async_list_frame_targets():
            label = self._frame_label(ti.url)
            try:
                html = await self._async_eval_in_frame_target(
                    ti.target_id, "document.body ? document.body.outerHTML : ''"
                )
            except Exception as e:
                trace(f"Extract from cross-origin frame {label} failed: {e}")
                continue
            if not html:
                trace(f"Cross-origin frame {label} has no body content")
                continue
            text = self.extract_text_with_links(html, ti.url)
            if text and len(text.strip()) > 50:
                fine(f"Extracted {len(text)} chars from cross-origin frame {label}")
                texts.append(f"[iframe {label}]\n{text}")
        return "\n\n".join(texts)

    async def _async_fetch_via_chrome(self, url: str) -> str | None:
        """Tier 1 fetch: navigate via nodriver, wait for DOM-quiet readiness, run cookie removal, scroll."""
        try:
            if not await self._async_ensure_browser():
                trace(f"fetch_via_chrome: browser unavailable for {url}")
                return None
            tab = self._tab

            try:
                await self._cdp_wait_for(tab.get(url), 30)
            except asyncio.TimeoutError:
                note(f"Page load timeout for {url}, stopping load and continuing...")
                await self._async_safe_stop_page_load(tab)
            except _BROWSER_DEAD_EXC as e:
                note(f"nodriver session broken ({type(e).__name__}: {e}) — retrying")
                await self._async_initialize_driver()
                if self._tab is None:
                    return None
                tab = self._tab
                try:
                    await self._cdp_wait_for(tab.get(url), 30)
                except asyncio.TimeoutError:
                    note(f"Page load timeout for {url} on retry, stopping load and continuing...")
                    await self._async_safe_stop_page_load(tab)

            await self._async_wait_for_page_ready(tab)
            await self._async_settle_cookie_banner(tab, url)
            await self._async_scroll_to_load_content(tab)
            return await self._async_regrab_if_blank(tab, url, await tab.get_content())
        except _BROWSER_DEAD_EXC as e:
            note(f"nodriver unhealthy after fetch for {url}: {e} — reinitializing")
            await self._async_initialize_driver()
            return None
        except Exception as e:
            trace(f"nodriver fetch failed for {url}: {e}")
            if not await self._async_browser_alive():
                note("fetch_via_chrome: opaque failure and the browser is dead — reinitializing")
                await self._async_initialize_driver()
            return None

    async def _async_regrab_if_blank(self, tab: 'nodriver.Tab', url: str, html: str) -> str:
        """Safety net under the readiness gate: a page whose markup renders no visible
        text is a shell still waiting for its data. Pause, let the network settle, and
        read the page once more — the second read is returned whatever it holds, so the
        embedded-JSON fallback of the extractor still has the shell to work on."""
        if self.extract_text_with_links(html or '', url, visible_only=True).strip():
            return html
        info(f"fetch_via_chrome: no visible text on {url} after the readiness gate — "
             f"re-reading the page once in {self._BLANK_REGRAB_WAIT_S:.0f}s")
        await asyncio.sleep(self._BLANK_REGRAB_WAIT_S)
        await self._async_wait_for_network_idle()
        regrab = await tab.get_content()
        visible = len(self.extract_text_with_links(regrab or '', url, visible_only=True).strip())
        if visible:
            info(f"fetch_via_chrome: the re-read of {url} found {visible} chars of visible text")
        else:
            trace(f"fetch_via_chrome: the re-read of {url} is still blank — using it as-is")
        return regrab

    def _fetch_via_chrome(self, url: str) -> str | None:
        """Sync wrapper around the nodriver Tier 1 fetch. Takes the browse lock — the
        fetch drives the same tab as the browse_* session, and interleaving the two
        (an open_url racing a browse command) wedges both."""
        try:
            return self._loop_thread.submit(
                self._with_browse_lock(self._async_fetch_via_chrome(url)), timeout=90)
        except Exception as e:
            trace(f"_fetch_via_chrome submit failed for {url}: {type(e).__name__}: {e}")
            return None

    async def _async_wait_for_driver(self, timeout: int = 30):
        """Block until the driver init coroutine signals readiness."""
        deadline = time.time() + timeout
        while not self._driver_ready.is_set():
            if time.time() > deadline:
                note("nodriver initialization timed out, forcing reinit")
                await self._async_initialize_driver()
                self._driver_ready.set()
                return
            await asyncio.sleep(0.1)

    @staticmethod
    def _is_pdf_url(url: str) -> bool:
        path = urlparse(url).path.lower()
        return path.endswith('.pdf')

    def _fetch_pdf_content(self, url: str) -> str:
        info(f"Detected PDF URL, downloading: {url}")
        try:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()

            content_length = response.headers.get('Content-Length')
            if content_length and int(content_length) > PdfExtractor.MAX_PDF_SIZE:
                note(f"PDF too large ({int(content_length)} bytes), skipping: {url}")
                return "ERROR: PDF file is too large (over 50 MB)."

            pdf_bytes = b''
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                pdf_bytes += chunk
                if len(pdf_bytes) > PdfExtractor.MAX_PDF_SIZE:
                    note(f"PDF download exceeded 50 MB limit: {url}")
                    return "ERROR: PDF file is too large (over 50 MB)."

            return PdfExtractor.extract_text(pdf_bytes)

        except requests.exceptions.RequestException as e:
            note(f"Failed to download PDF {url}: {e}")
            return f"ERROR: Could not download PDF: {e}"

    async def _async_fetch_pdf_bytes_via_chrome(self, url: str):
        """Download PDF bytes through Chrome's network stack so cookies, redirects,
        and anti-bot protections from the live browse session apply.
        Returns raw bytes on success, or an 'ERROR: ...' string on failure."""
        if self._tab is None:
            return "ERROR: browser is not available."

        info(f"browse_open: downloading PDF via Chrome: {url}")

        try:
            frame_tree = await self._tab.send(cdp.page.get_frame_tree())
            frame_id = frame_tree.frame.id_
        except Exception as e:
            trace(f"PDF chrome download: get_frame_tree failed: {e}")
            return f"ERROR: could not resolve page frame: {e}"

        try:
            result = await self._tab.send(cdp.network.load_network_resource(
                url=url,
                options=cdp.network.LoadNetworkResourceOptions(
                    disable_cache=False,
                    include_credentials=True,
                ),
                frame_id=frame_id,
            ))
        except Exception as e:
            trace(f"PDF chrome download: loadNetworkResource failed: {e}")
            return f"ERROR: Chrome download failed: {e}"

        if not result.success:
            err_name = result.net_error_name or (
                f"HTTP {int(result.http_status_code)}" if result.http_status_code else "unknown"
            )
            note(f"PDF chrome download failed for {url}: {err_name}")
            return f"ERROR: Chrome could not download PDF ({err_name})."

        if result.stream is None:
            note(f"PDF chrome download: no stream handle returned for {url}")
            return "ERROR: Chrome download returned no stream."

        handle = result.stream
        pdf_bytes = bytearray()
        try:
            while True:
                base64_encoded, data, eof = await self._tab.send(
                    cdp.io.read(handle=handle, size=1024 * 256)
                )
                if data:
                    chunk = base64.b64decode(data) if base64_encoded else data.encode('latin-1')
                    pdf_bytes.extend(chunk)
                    if len(pdf_bytes) > PdfExtractor.MAX_PDF_SIZE:
                        note(f"PDF chrome download exceeded 50 MB limit: {url}")
                        return "ERROR: PDF file is too large (over 50 MB)."
                if eof:
                    break
        except Exception as e:
            trace(f"PDF chrome download: stream read failed: {e}")
            return f"ERROR: Chrome stream read failed: {e}"
        finally:
            try:
                await self._tab.send(cdp.io.close(handle=handle))
            except Exception as e:
                trace(f"PDF chrome download: stream close failed: {e}")

        if not pdf_bytes.startswith(b'%PDF-'):
            note(f"PDF chrome download: response not a PDF (first bytes: {bytes(pdf_bytes[:8])!r}) for {url}")
            return "ERROR: Chrome downloaded content but it is not a PDF (likely an auth or error page)."

        info(f"PDF chrome download: got {len(pdf_bytes)} bytes for {url}")
        return bytes(pdf_bytes)

    # ------------------------------------------------------------------
    # Interactive browse session
    # ------------------------------------------------------------------

    # Why an <iframe> shows nothing, '' when it renders. Shared by the collector
    # (the page's own iframes) and the per-target pre-pass (an iframe's children):
    # the frame's own style and box, then its ancestors — a frame whose own style
    # says display:block still shows nothing inside an opacity-0 wrapper (Kustom's
    # wallet buttons), when an overflow-clipping ancestor's box does not reach it,
    # or inside a collapsed 0px row (Kustom's unselected "Betala med kort" card
    # form: the row has overflow visible, the form spills out of it and the rows
    # below paint on top). The last case is settled by hit-testing the frame's
    # centre when it is on screen — a 0px float container is the same shape with
    # the frame painted — and is taken as collapsed when it is not. Inline
    # ancestors have no box of their own and are skipped.
    _IFRAME_HIDDEN_BY_JS = r"""
    function iframePainted(f, win) {
        // true/false when the frame's centre is on screen, null when it is not.
        const r = f.getBoundingClientRect();
        const x = r.left + r.width / 2, y = r.top + r.height / 2;
        if (x < 0 || y < 0 || x >= win.innerWidth || y >= win.innerHeight) return null;
        try { return f.ownerDocument.elementFromPoint(x, y) === f; } catch (e) { return null; }
    }

    function iframeHiddenBy(f, win) {
        const style = win.getComputedStyle(f);
        if (style.display === 'none') return 'display:none';
        if (style.visibility === 'hidden') return 'visibility:hidden';
        const rect = f.getBoundingClientRect();
        if (!(rect.width > 10 && rect.height > 10)) return Math.round(rect.width) + 'x' + Math.round(rect.height);
        const body = f.ownerDocument.body;
        for (let p = f.parentElement; p && p !== body; p = p.parentElement) {
            const ps = win.getComputedStyle(p);
            const tag = p.tagName.toLowerCase();
            if (parseFloat(ps.opacity) === 0) return 'inside opacity:0 ' + tag;
            if (ps.display === 'inline' || ps.display === 'contents') continue;
            const pr = p.getBoundingClientRect();
            if (pr.height < 2) {
                const painted = iframePainted(f, win);
                if (painted === true) return '';
                return 'inside collapsed ' + tag + (painted === false ? ', covered' : '');
            }
            if (ps.overflowX === 'visible' && ps.overflowY === 'visible') continue;
            const w = Math.min(rect.right, pr.right) - Math.max(rect.left, pr.left);
            const h = Math.min(rect.bottom, pr.bottom) - Math.max(rect.top, pr.top);
            if (w <= 10 || h <= 10) return 'clipped by ' + tag + ' ' + Math.round(pr.width) + 'x' + Math.round(pr.height);
        }
        return '';
    }
"""

    _BROWSE_COLLECT_JS = r"""
(function(startIdx, sourceLabel) {
    // Walk this document and (best-effort) any same-origin iframes, tagging each
    // interactive element with data-floxy-snap-idx so Python can re-resolve it.
    // Runs once in the main frame (startIdx 0) and once per out-of-process
    // cross-origin iframe target with an index offset, so indices are globally
    // unique across the merged snapshot.
    function clearTags(doc) {
        try {
            doc.querySelectorAll('[data-floxy-snap-idx]').forEach(e => e.removeAttribute('data-floxy-snap-idx'));
        } catch (e) {}
    }

    function isInteractive(el, win) {
        if (!el || el.disabled) return false;
        if (el.getAttribute('aria-disabled') === 'true') return false;
        const cls = (el.className && el.className.toString) ? el.className.toString().toLowerCase() : '';
        if (/(^|[\s_-])(is-)?disabled([\s_-]|$)/.test(cls)) return false;
        const style = win.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden' || parseFloat(style.opacity) === 0) return false;
        if (style.pointerEvents === 'none') return false;
        const rect = el.getBoundingClientRect();
        if (rect.width < 2 || rect.height < 2) return false;
        // Reject elements positioned entirely outside the document (e.g. Shopify's
        // PCI iframes hide autofill-shadow card fields at left:-9999px — they must
        // stay technically visible so browser autofill fills them). Compare in
        // document coordinates: rects are viewport-relative, and below-the-fold
        // elements merely scrolled out of view must survive this check.
        const docX = rect.left + win.scrollX;
        const docY = rect.top + win.scrollY;
        if (docX + rect.width <= 0 || docY + rect.height <= 0) return false;
        const docEl = el.ownerDocument.documentElement;
        if (docX >= Math.max(docEl.scrollWidth, docEl.clientWidth)) return false;
        if (docY >= Math.max(docEl.scrollHeight, docEl.clientHeight)) return false;
        return true;
    }

    function hiddenChoiceHost(el, win) {
        // A radio or checkbox parked off-screen (sr-only, 1x1) behind a styled
        // <label> — Kustom's payment methods, most card pickers. The label is
        // what the user clicks, so it takes the index; the record still
        // describes the input (type, checked).
        if (el.tagName.toLowerCase() !== 'input' || el.disabled) return null;
        const type = (el.getAttribute('type') || '').toLowerCase();
        if (type !== 'radio' && type !== 'checkbox') return null;
        let host = el.closest('label');
        if (!host && el.id) {
            try { host = el.ownerDocument.querySelector('label[for="' + CSS.escape(el.id) + '"]'); } catch (e) {}
        }
        return host && isInteractive(host, win) ? host : null;
    }

/*IFRAME_HIDDEN_BY*/

    function ancestorContext(el, ownLabel) {
        const HEADING_SEL = 'h1, h2, h3, h4, h5, h6, legend, [class*="heading" i], [class*="title" i], [class*="month" i][class*="head" i]';
        let p = el.parentElement;
        let depth = 0;
        let rowHeader = '';
        while (p && depth < 8) {
            const cls = (p.className && p.className.toString) ? p.className.toString().toLowerCase() : '';
            if (/heading|title|month|section|group|legend|panel|card|block/.test(cls) ||
                p.tagName === 'SECTION' || p.tagName === 'FIELDSET' || p.tagName === 'FORM' ||
                p.tagName === 'TABLE') {
                const heading = p.querySelector(HEADING_SEL);
                if (heading) {
                    const txt = (heading.innerText || '').trim().split('\n')[0].trim();
                    if (txt && txt.length <= 50) return txt;
                }
            }
            if (!rowHeader) {
                const ptxt = (p.innerText || '').trim();
                if (ptxt) {
                    const firstLine = ptxt.split('\n')[0].trim();
                    if (firstLine && firstLine.length <= 40 &&
                        firstLine.toLowerCase() !== ownLabel.toLowerCase() &&
                        !firstLine.toLowerCase().endsWith(ownLabel.toLowerCase())) {
                        rowHeader = firstLine;
                    }
                }
            }
            p = p.parentElement;
            depth++;
        }
        return rowHeader;
    }

    function label(el) {
        const aria = el.getAttribute('aria-label');
        if (aria && aria.trim()) return aria.trim();
        const labelledBy = el.getAttribute('aria-labelledby');
        if (labelledBy) {
            const ref = el.ownerDocument.getElementById(labelledBy);
            if (ref && ref.innerText && ref.innerText.trim()) return ref.innerText.trim();
        }
        if (el.id) {
            const lbl = el.ownerDocument.querySelector('label[for="' + CSS.escape(el.id) + '"]');
            if (lbl && lbl.innerText && lbl.innerText.trim()) return lbl.innerText.trim();
        }
        // A wrapping <label> names the control it wraps, not a link or button
        // that merely sits inside it (Kustom's payment rows are labels holding a
        // "köparskydd" link — that link keeps its own text).
        const tagName = el.tagName.toLowerCase();
        if (tagName === 'input' || tagName === 'select' || tagName === 'textarea') {
            const parentLabel = el.closest('label');
            if (parentLabel && parentLabel.innerText && parentLabel.innerText.trim()) {
                return parentLabel.innerText.trim();
            }
        }
        const txt = (el.innerText || el.textContent || '').trim();
        if (txt) return txt;
        const ph = el.getAttribute('placeholder');
        if (ph && ph.trim()) return ph.trim();
        const name = el.getAttribute('name');
        if (name) return name;
        const title = el.getAttribute('title');
        if (title) return title;
        return '';
    }

    const SEL = [
        'input', 'select', 'textarea', 'button', 'a[href]',
        '[role="button"]', '[role="link"]', '[role="tab"]', '[role="checkbox"]', '[role="radio"]',
        '[role="combobox"]', '[role="textbox"]', '[role="menuitem"]', '[role="option"]',
        '[contenteditable=""]', '[contenteditable="true"]',
        '[onclick]', '[ng-click]', '[data-ng-click]', '[ng-submit]',
    ].join(', ');

    // Handler-bound controls: widgets built from plain divs/spans (booking
    // calendars, quantity pickers, tab strips, card pickers) bind clicks with
    // addEventListener/jQuery and carry no role, href or onclick attribute, so
    // SEL never sees them and the LLM ends up reverse-engineering the DOM with
    // browse_eval. Two signals catch them: a computed cursor:pointer, or a
    // choice-naming data-* attribute. Only the outermost pointer element counts
    // (cursor inherits; the container is the click target), never one that
    // wraps or sits inside a SEL control, and never a big container.
    const CHOICE_ATTRS = ['data-value', 'data-qty', 'data-date', 'data-time', 'data-option',
                          'data-step', 'data-tab', 'data-action', 'data-id'];
    const HANDLER_TAGS = { div: 1, span: 1, li: 1, td: 1, p: 1, label: 1, img: 1, i: 1, summary: 1, svg: 1 };
    const HANDLER_MAX_DESCENDANTS = 25;
    const HANDLER_MAX_TEXT = 160;
    const HANDLER_MAX_RECORDS = 150;

    function choiceHint(el) {
        for (const a of CHOICE_ATTRS) {
            const v = el.getAttribute(a);
            if (v !== null && v !== '') return a + '=' + v.slice(0, 40);
        }
        return '';
    }

    function handlerBound(doc, win) {
        const found = [];
        let candidates;
        try { candidates = doc.querySelectorAll('body *'); } catch (e) { return found; }
        for (const el of candidates) {
            if (found.length >= HANDLER_MAX_RECORDS) break;
            const tag = el.tagName.toLowerCase();
            if (!HANDLER_TAGS[tag]) continue;
            const choice = choiceHint(el);
            let pointer = false;
            try { pointer = win.getComputedStyle(el).cursor === 'pointer'; } catch (e) {}
            if (!pointer && !choice) continue;
            // data-id alone is too common (list rows, tracking) — it needs the cursor too.
            if (!pointer && choice.indexOf('data-id=') === 0) continue;
            try {
                if (el.closest(SEL) || el.querySelector(SEL)) continue;
            } catch (e) { continue; }
            if (pointer && !choice) {
                const parent = el.parentElement;
                let parentPointer = false;
                try { parentPointer = parent && parent !== doc.body && win.getComputedStyle(parent).cursor === 'pointer'; } catch (e) {}
                if (parentPointer) continue;
            }
            if (el.getElementsByTagName('*').length > HANDLER_MAX_DESCENDANTS) continue;
            const raw = (el.innerText || el.textContent || '').trim();
            if (raw.length > HANDLER_MAX_TEXT) continue;
            if (!raw && !choice && !el.getAttribute('aria-label') && !el.getAttribute('title')) continue;
            found.push(el);
        }
        return found;
    }

    function collect(doc, win, sourceLabel) {
        const els = Array.from(doc.querySelectorAll(SEL));
        const handlerEls = handlerBound(doc, win);
        const handlerSet = new Set(handlerEls);
        for (const el of handlerEls) els.push(el);
        const records = [];
        for (const el of els) {
            let target = el;
            if (!isInteractive(el, win)) {
                target = hiddenChoiceHost(el, win);
                if (!target) continue;
            }
            const tag = el.tagName.toLowerCase();
            const role = el.getAttribute('role') || '';
            const type = (el.getAttribute('type') || '').toLowerCase();
            let lbl = label(el).slice(0, 120).replace(/\s+/g, ' ');
            if (tag === 'a' && !lbl) continue;
            if (tag === 'a' && (el.getAttribute('href') || '').trim() === '#') continue;

            if (lbl.length <= 3 || /^\d+$/.test(lbl)) {
                const ctx = ancestorContext(el, lbl);
                if (ctx) lbl = ctx + ' / ' + lbl;
            }

            const rec = { tag, type, role, label: lbl };
            if (sourceLabel) rec.source = sourceLabel;
            if (handlerSet.has(el)) {
                // State usually lives in the class list on such widgets
                // (day av / wait / closed, time waitlist, tab active) — pass it on.
                rec.clickable = true;
                const hint = choiceHint(el);
                if (hint) rec.data = hint;
                const cls = (el.getAttribute('class') || '').trim().replace(/\s+/g, ' ');
                if (cls) rec.cls = cls.slice(0, 40);
            }

            if ((tag === 'input' || tag === 'textarea') &&
                !['checkbox','radio','submit','button','reset','hidden','file','image',
                  'range','color','date','time','datetime-local','month','week'].includes(type)) {
                const hay = lbl + ' ' + (el.getAttribute('name') || '') + ' ' +
                            (el.getAttribute('placeholder') || '') + ' ' +
                            (el.getAttribute('aria-label') || '');
                if (type === 'search' || /search|sök/i.test(hay)) rec.isSearch = true;
            }

            if (tag === 'input' || tag === 'textarea' || el.isContentEditable) {
                rec.value = el.value !== undefined ? el.value : (el.innerText || '');
                if (type === 'checkbox' || type === 'radio') rec.checked = el.checked;
            } else if (tag === 'select') {
                rec.value = el.value;
                rec.options = Array.from(el.options).map(o => ({ value: o.value, text: (o.text || '').trim() }));
            } else if (tag === 'a') {
                rec.href = el.getAttribute('href') || '';
            }
            records.push({ rec, el: target });
        }
        return records;
    }

    function dedupNested(all) {
        // A container and a nested interactive element often carry the same
        // label (a menu li plus its button) — drop the container so each
        // control is listed once. Only equal normalized labels are considered,
        // and the more semantic element wins (containers lose ties).
        const SEMANTIC = { button: 3, a: 3, input: 3, select: 3, textarea: 3 };
        const byEl = new Map();
        for (const r of all) byEl.set(r.el, r);
        const drop = new Set();
        const norm = s => (s || '').trim().toLowerCase();
        for (const r of all) {
            const lbl = norm(r.rec.label);
            if (!lbl) continue;
            let p = r.el.parentElement;
            while (p) {
                const outer = byEl.get(p);
                if (outer && !drop.has(outer) && norm(outer.rec.label) === lbl) {
                    const innerScore = SEMANTIC[r.rec.tag] || 0;
                    const outerScore = SEMANTIC[outer.rec.tag] || 0;
                    if (innerScore >= outerScore) drop.add(outer); else drop.add(r);
                }
                p = p.parentElement;
            }
        }
        return all.filter(r => !drop.has(r));
    }

    function regionOf(el) {
        let zone = null;
        try {
            zone = el.closest('nav, header, footer, aside, [role="navigation"], [role="banner"], [role="contentinfo"]');
        } catch (e) {}
        if (!zone) return 'main';
        const t = zone.tagName.toLowerCase();
        if (t === 'nav' || t === 'header' || t === 'footer' || t === 'aside') return t;
        const role = zone.getAttribute('role');
        if (role === 'navigation') return 'nav';
        if (role === 'banner') return 'header';
        return 'footer';
    }

    function pageContext(doc) {
        const ctx = {};
        try {
            const h1 = doc.querySelector('h1');
            if (h1) {
                const t = (h1.innerText || '').trim().replace(/\s+/g, ' ').slice(0, 80);
                if (t) ctx.h1 = t;
            }
            const headings = [];
            const seen = {};
            for (const h of doc.querySelectorAll('h2, h3')) {
                if (h.closest('nav, header, footer, aside')) continue;
                const t = (h.innerText || '').trim().replace(/\s+/g, ' ').slice(0, 60);
                if (!t || seen[t.toLowerCase()]) continue;
                seen[t.toLowerCase()] = true;
                headings.push(t);
                if (headings.length >= 4) break;
            }
            if (headings.length) ctx.headings = headings;
        } catch (e) {}
        try {
            // Status messages: what the page is telling the user right now (an error
            // under a form, a "we sent you a code" banner, a toast).
            const alerts = [];
            const seenAlert = {};
            const ALERT_SEL = '[role="alert"], [role="status"], [aria-live="polite"], [aria-live="assertive"], ' +
                '.alert, .error, .warning, .notice, .toast, .snackbar, .mat-error, .help-block, ' +
                '[class*="error-message" i], [class*="errormessage" i], [class*="form-error" i], ' +
                '[class*="alert-" i], [class*="notification" i], [class*="validation" i]';
            const title = (doc.title || '').toLowerCase();
            const h1Text = (ctx.h1 || '').toLowerCase();
            for (const el of doc.querySelectorAll(ALERT_SEL)) {
                if (el.closest('nav, header, footer, aside, script, style, template')) continue;
                if (!el.getClientRects().length) continue;
                const t = (el.innerText || '').trim().replace(/\s+/g, ' ');
                if (!t || t.length > 300 || seenAlert[t.toLowerCase()]) continue;
                // A one-word "alert" is a badge; a heading echoed through an aria-live
                // region is not a status message.
                if (t.indexOf(' ') === -1) continue;
                const lt = t.toLowerCase();
                if (lt === h1Text || (title && title.indexOf(lt) !== -1)) continue;
                if (alerts.some(a => a.indexOf(t) !== -1 || t.indexOf(a) !== -1)) continue;
                seenAlert[t.toLowerCase()] = true;
                alerts.push(t.slice(0, 200));
                if (alerts.length >= 5) break;
            }
            if (alerts.length) ctx.alerts = alerts;
        } catch (e) {}
        try {
            // A digest of the visible main-content text, so a page whose meaning is in
            // its prose (a 2FA instruction, a confirmation, a "no tables left" line) is
            // understood from the snapshot without a browse_extract round. Text nodes
            // in document order, chrome (nav/header/footer/aside) and hidden subtrees
            // skipped, cut at the budget — cheap because the walk stops there.
            const BUDGET = 500;
            const SKIP = 'nav, header, footer, aside, script, style, noscript, template, svg, ' +
                '[aria-hidden="true"], [role="navigation"], [role="banner"], [role="contentinfo"]';
            function digest(root) {
                const hidden = new Set();
                const walker = doc.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
                    acceptNode: function(node) {
                        const t = node.nodeValue.trim();
                        if (!t) return NodeFilter.FILTER_REJECT;
                        let el = node.parentElement;
                        if (!el) return NodeFilter.FILTER_REJECT;
                        // Chrome and aria-hidden subtrees are skipped inside the root only:
                        // a modal marks everything outside itself aria-hidden.
                        for (let p = el; p && p !== root; p = p.parentElement) {
                            if (hidden.has(p)) return NodeFilter.FILTER_REJECT;
                            if (p.matches(SKIP)) { hidden.add(p); return NodeFilter.FILTER_REJECT; }
                        }
                        if (!el.getClientRects().length) { hidden.add(el); return NodeFilter.FILTER_REJECT; }
                        return NodeFilter.FILTER_ACCEPT;
                    }
                });
                const parts = [];
                let total = 0;
                let visited = 0;
                let node;
                while ((node = walker.nextNode()) && visited++ < 4000) {
                    const t = node.nodeValue.trim().replace(/\s+/g, ' ');
                    parts.push(t);
                    total += t.length + 1;
                    if (total >= BUDGET) break;
                }
                let text = parts.join(' ');
                if (text.length > BUDGET) text = text.slice(0, BUDGET).replace(/\s+\S*$/, '') + ' …';
                else if (node && visited >= 4000) text += ' …';
                return text;
            }
            // An open modal is what the user is looking at — and Angular/CDK marks the
            // rest of the page aria-hidden while it is up, so `main` would yield nothing.
            const dialogs = Array.from(doc.querySelectorAll(
                '[role="dialog"], [role="alertdialog"], [aria-modal="true"], dialog[open]'))
                .filter(d => d.getClientRects().length);
            let text = '';
            if (dialogs.length) {
                text = digest(dialogs[dialogs.length - 1]);
                if (text) text = '(dialog) ' + text;
            }
            if (!text) text = digest(doc.querySelector('main, [role="main"], article') || doc.body);
            if (!text && doc.body) text = digest(doc.body);
            if (text) ctx.text = text;
        } catch (e) {}
        return ctx;
    }

    clearTags(document);

    let all = collect(document, window, sourceLabel || '');

    // Walk same-origin iframes; record cross-origin iframe srcs separately so the
    // caller can surface them (the snapshot lists the ones it cannot read).
    const skipIframePatterns = ['maps.google.com', 'google.com/maps', 'cookiebot.com'];
    const crossOriginFrames = [];
    const hiddenFrames = [];
    const iframes = Array.from(document.querySelectorAll('iframe'));
    for (let i = 0; i < iframes.length; i++) {
        const f = iframes[i];
        const rawSrc = f.getAttribute('src') || '';
        if (skipIframePatterns.some(p => rawSrc.indexOf(p) !== -1)) continue;
        let doc = null;
        let win = null;
        try { doc = f.contentDocument; win = f.contentWindow; } catch (e) {}
        if (!doc || !win) {
            // Cross-origin (or otherwise inaccessible) iframe with a usable URL.
            // Only drop tracking pixels and the like: a Stripe card field is a
            // ~17px-tall iframe, so the threshold must stay far below that.
            // The collapsed ones are reported too: their contents look visible
            // from inside (a 0px-tall Stripe autofill frame still holds a
            // "Kortnummer" input) and the snapshot must not list them.
            if (!rawSrc) continue;
            let abs = rawSrc;
            try { abs = f.src || rawSrc; } catch (e) {}
            const why = iframeHiddenBy(f, window);
            if (why) hiddenFrames.push({ src: abs, why }); else crossOriginFrames.push(abs);
            continue;
        }
        try { clearTags(doc); } catch (e) { continue; }
        const childLabel = rawSrc.split('?')[0] || `iframe[${i}]`;
        const found = collect(doc, win, sourceLabel ? sourceLabel + ' > ' + childLabel : childLabel);
        all = all.concat(found);
    }

    // De-dup nested duplicates BEFORE index tagging so data-floxy-snap-idx stays
    // dense and the Python frame-id/label tables stay aligned, then tag regions.
    all = dedupNested(all);
    for (const r of all) r.rec.region = regionOf(r.el);

    // Disambiguate duplicate labels with ancestor context (skip already-disambiguated)
    const counts = {};
    for (const r of all) counts[r.rec.label] = (counts[r.rec.label] || 0) + 1;
    for (const r of all) {
        if (counts[r.rec.label] > 1 && r.rec.label.indexOf(' / ') === -1) {
            const ctx = ancestorContext(r.el, r.rec.label);
            if (ctx) r.rec.label = ctx + ' / ' + r.rec.label;
        }
    }

    // Tag and produce the final records array
    const records = [];
    all.forEach((r, i) => {
        try { r.el.setAttribute('data-floxy-snap-idx', String(startIdx + i)); } catch (e) {}
        records.push(r.rec);
    });
    return { records, crossOriginFrames, hiddenFrames, context: pageContext(document) };
})""".replace('/*IFRAME_HIDDEN_BY*/', _IFRAME_HIDDEN_BY_JS)

    # Runs inside an iframe target before the collector: which cross-origin iframes
    # it hosts, split by whether they render. A nested frame (Stripe's helpers inside
    # a Kustom checkout) is only known to be collapsed by its parent — from inside,
    # a 0px-tall frame's inputs pass every visibility test.
    _FRAME_CHILD_IFRAMES_JS = r"""(() => {
/*IFRAME_HIDDEN_BY*/
    const out = { shown: [], hidden: [] };
    for (const f of document.querySelectorAll('iframe')) {
        let src = '';
        try { src = f.src || f.getAttribute('src') || ''; } catch (e) {}
        if (!src) continue;
        const style = getComputedStyle(f);
        const rect = f.getBoundingClientRect();
        const why = iframeHiddenBy(f, window);
        (why ? out.hidden : out.shown).push(
            { src, w: Math.round(rect.width), h: Math.round(rect.height), display: style.display,
              visibility: style.visibility, why });
    }
    return out;
})()""".replace('/*IFRAME_HIDDEN_BY*/', _IFRAME_HIDDEN_BY_JS)

    # Wording coordinated with the browse_act \n suffix / browse_key command docs.
    _SEARCH_FIELD_HINT = (' (search field: filling does not submit — '
                          'fill with trailing \\n or browse_key("Enter") to submit)')

    def _format_snapshot_line(self, idx: int, rec: dict) -> str:
        tag = rec.get('tag', '')
        type_ = rec.get('type', '')
        role = rec.get('role', '')
        label_ = rec.get('label', '') or '(no label)'

        if tag == 'input' or tag == 'textarea':
            kind = f'input[{type_}]' if type_ else tag
            extra = ''
            if type_ in ('checkbox', 'radio'):
                extra = f' checked={rec.get("checked", False)}'
            else:
                v = rec.get('value', '')
                if v:
                    extra = f' value={v!r}'
                if type_ == 'search' or rec.get('isSearch'):
                    extra += self._SEARCH_FIELD_HINT
            return f'[{idx}] {kind} "{label_}"{extra}'

        if tag == 'select':
            opts = rec.get('options', [])
            opt_str = ','.join(o.get('text') or o.get('value') or '' for o in opts)
            return f'[{idx}] select "{label_}" value={rec.get("value", "")!r} options=[{opt_str}]'

        if tag == 'button' or role == 'button':
            return f'[{idx}] button "{label_}"'

        if tag == 'a' or role == 'link':
            href = rec.get('href', '')
            return f'[{idx}] link "{label_}" -> {href}'

        if rec.get('contenteditable') or role == 'textbox':
            v = rec.get('value', '')
            return f'[{idx}] editable "{label_}" value={v!r}'

        if rec.get('clickable'):
            hints = [h for h in (rec.get('data'), rec.get('cls') and f'class="{rec["cls"]}"') if h]
            return f'[{idx}] clickable {tag} "{label_}"' + (f' ({", ".join(hints)})' if hints else '')

        return f'[{idx}] {tag} "{label_}"'

    # A nav/header/footer/aside group larger than this renders as one compact
    # line instead of one line per element (indices stay visible and clickable).
    _SNAPSHOT_GROUP_COLLAPSE_THRESHOLD = 5
    _SNAPSHOT_REGION_ORDER = ('header', 'nav', 'aside', 'footer')
    _SNAPSHOT_GROUP_LABEL_MAX = 40

    @staticmethod
    def _format_page_context(context) -> str:
        """Render the collector's page context ({h1, headings, alerts, text}) as the
        'Page:' / 'Alerts:' / 'Text:' header lines."""
        if not isinstance(context, dict):
            return ''
        parts = []
        h1 = context.get('h1')
        if h1:
            parts.append(f'h1 "{h1}"')
        headings = context.get('headings') or []
        if headings:
            parts.append('headings: ' + ', '.join(f'"{h}"' for h in headings))
        out = f"Page: {' | '.join(parts)}\n" if parts else ''
        alerts = context.get('alerts') or []
        if alerts:
            out += 'Alerts: ' + ' | '.join(f'"{a}"' for a in alerts) + '\n'
        text = context.get('text')
        if text:
            out += f"{SnapshotFormat.TEXT_PREFIX}{text}\n"
        return out

    def _render_snapshot_lines(self, records: list) -> list[str]:
        """Grouped display rendering: main-content records first, then nav/header/
        footer/aside groups after a separator, large groups collapsed to one line.

        Purely display-order — indices are the records' positions exactly as the
        collector tagged them, so _browse_snapshot_frame_ids/_labels stay aligned."""
        def line_for(i):
            line = self._format_snapshot_line(i, records[i])
            src = records[i].get('source')
            if src:
                line += f'  [in {src}]'
            return line

        groups: dict[str, list[int]] = {}
        main_lines = []
        for i, rec in enumerate(records):
            region = rec.get('region') or 'main'
            if region == 'main':
                main_lines.append(line_for(i))
            else:
                groups.setdefault(region, []).append(i)

        lines = main_lines
        if groups:
            lines = lines + ([SnapshotFormat.GROUP_SEPARATOR] if not lines
                             else ['', SnapshotFormat.GROUP_SEPARATOR])
            ordered = [r for r in self._SNAPSHOT_REGION_ORDER if r in groups]
            ordered += [r for r in groups if r not in self._SNAPSHOT_REGION_ORDER]
            for region in ordered:
                idxs = groups[region]
                if len(idxs) <= self._SNAPSHOT_GROUP_COLLAPSE_THRESHOLD:
                    lines.extend(line_for(i) for i in idxs)
                else:
                    items = []
                    for i in idxs:
                        label = (records[i].get('label') or '').strip() or '(no label)'
                        if len(label) > self._SNAPSHOT_GROUP_LABEL_MAX:
                            label = label[:self._SNAPSHOT_GROUP_LABEL_MAX - 1] + '…'
                        kind = self._collapsed_item_kind(records[i])
                        items.append(f'[{i}] {kind + " " if kind else ""}{label}')
                    lines.append(f'{region}: {len(idxs)} items — ' + ', '.join(items))
        return lines

    @staticmethod
    def _collapsed_item_kind(rec: dict) -> str:
        """Compact type tag for a collapsed-group item. Links are the overwhelming
        default in header/nav groups and stay bare; anything the LLM could mistake
        for a text field (buttons, inputs, selects) gets labeled — a bare
        '[2] Sök efter produkter' once read as a search input and drew three
        doomed fill attempts at a button."""
        tag = rec.get('tag', '')
        role = rec.get('role', '')
        if tag == 'a' or role == 'link':
            return ''
        if tag in ('input', 'textarea'):
            type_ = rec.get('type', '')
            return f'input[{type_}]' if type_ else tag
        if tag == 'button' or role == 'button':
            return 'button'
        if rec.get('contenteditable') or role == 'textbox':
            return 'editable'
        if rec.get('clickable'):
            return f'clickable {tag}'
        return tag

    @staticmethod
    def _urls_differ(requested: str, final: str) -> bool:
        """True when the final URL is a materially different page from the requested
        one (host ignoring www., path ignoring trailing slash). Query strings and
        scheme upgrades are ignored to keep benign redirects quiet."""
        def key(u):
            p = urlparse(u or '')
            host = (p.netloc or '').lower()
            if host.startswith('www.'):
                host = host[4:]
            path = (p.path or '/').rstrip('/') or '/'
            return host, path
        try:
            return key(requested) != key(final)
        except Exception:
            return False

    async def _async_take_snapshot(self, requested_url: Optional[str] = None) -> str:
        """Run the collector JS in the main frame and each cross-origin iframe target,
        merge into one globally-indexed snapshot, and store the index→frame routing table."""
        await self._async_sync_popup_targets()
        if self._tab is None:
            return "ERROR: browser not available."
        # The snapshot renders each link's href as the page wrote it, so a root-relative
        # path the LLM hands back must resolve against the page the snapshot shows —
        # the tab's, after any redirect or in-page navigation — not the last fetched one.
        self._anchor_links_to_tab(requested_url)
        try:
            result = await self._evaluate(self._tab, f"({self._BROWSE_COLLECT_JS})(0, '')")
        except Exception as e:
            trace(f"snapshot: collector JS failed: {type(e).__name__}: {e}")
            return "ERROR: snapshot collector failed."

        if not result or not isinstance(result, dict):
            self._browse_snapshot_frame_ids = []
            self._browse_snapshot_labels = []
            return SnapshotFormat.NO_ELEMENTS

        records = result.get('records', []) or []
        cross_origin_srcs = result.get('crossOriginFrames', []) or []
        context = result.get('context')
        frame_ids: list = [None] * len(records)

        # Cross-origin iframes are separate CDP targets — run the collector inside
        # each one via an attached session, offsetting indices to stay globally unique.
        # The whole pass shares one time budget: a page full of dead/slow OOPIFs (a
        # Shopify checkout has ~8 PCI frames) must degrade to a main-frame-only
        # snapshot, not stack per-frame timeouts until the watchdog kills the command.
        attached_labels = set()
        frame_targets = await self._async_settled_frame_targets(cross_origin_srcs)
        frame_deadline = asyncio.get_running_loop().time() + self._SNAPSHOT_FRAME_BUDGET
        hidden_urls, hidden_labels = await self._async_hidden_frames(result, frame_targets, frame_deadline)
        skipped_hidden: list[str] = []
        for n, ti in enumerate(frame_targets):
            label = self._frame_label(ti.url)
            if self._normalize_url(ti.url) in hidden_urls or label in hidden_labels:
                skipped_hidden.append(label)
                continue
            if asyncio.get_running_loop().time() > frame_deadline:
                note(f"Snapshot: OOPIF budget ({self._SNAPSHOT_FRAME_BUDGET:.0f}s) spent — "
                     f"skipping the remaining {len(frame_targets) - n} frame(s), starting with {label}")
                break
            offset = len(records)
            try:
                fres = await self._async_eval_in_frame_target(
                    ti.target_id,
                    f"({self._BROWSE_COLLECT_JS})({offset}, '{self._js_escape(label)}')",
                )
            except Exception as e:
                note(f"Snapshot: collecting from cross-origin frame {label} failed: "
                     f"{type(e).__name__}: {e}")
                continue
            frame_records = (fres.get('records') or []) if isinstance(fres, dict) else []
            attached_labels.add(label)
            if not frame_records:
                trace(f"Snapshot: cross-origin frame {label} has no interactive elements")
                continue
            info(f"Snapshot: {len(frame_records)} elements from cross-origin frame {label}")
            records.extend(frame_records)
            frame_ids.extend([str(ti.target_id)] * len(frame_records))

        # Frames whose srcs were seen in the page but whose targets we could not
        # reach stay hidden — keep reporting those so the LLM knows they exist.
        # Matched per frame (host + path), not per host: Stripe mounts several
        # js.stripe.com frames and reaching the controller must not hide the
        # card field.
        hidden_frames = [
            s for s in cross_origin_srcs
            if self._frame_label(self._normalize_url(s)) not in attached_labels
        ]

        self._browse_snapshot_frame_ids = frame_ids
        self._browse_snapshot_labels = [(r.get('label') or '').strip() for r in records]

        url = ''
        title = ''
        try:
            url = self._tab.target.url or ''
        except Exception:
            pass
        try:
            title = await self._evaluate(self._tab, "document.title") or ''
        except Exception:
            pass

        lines = self._render_snapshot_lines(records)

        info(f"Snapshot: {len(records)} interactive elements at {url}")
        if skipped_hidden:
            info(f"Snapshot: left out {len(skipped_hidden)} collapsed/hidden iframe target(s) "
                 f"whose contents the page does not show: {sorted(set(skipped_hidden))}")
        if hidden_frames:
            info(f"Snapshot: {len(hidden_frames)} cross-origin iframe(s) "
                 f"with hidden contents: {hidden_frames}")
        trace('\n'.join(lines))

        warning = ''
        if requested_url and url and self._urls_differ(requested_url, url):
            note(f"snapshot: final URL {url} differs from the requested {requested_url}")
            warning = (f"WARNING: the browser ended up on {url} — NOT the requested "
                       f"{requested_url} (silent redirect?). The snapshot below shows the actual page.\n")

        return self._assemble_snapshot(self._consume_target_notes() + warning, url, title,
                                       self._format_page_context(context), lines, hidden_frames)

    @staticmethod
    def _assemble_snapshot(notes: str, url: str, title: str, page_context: str, lines: list[str],
                           hidden_frames: list[str]) -> str:
        """The snapshot text in the shape SnapshotFormat documents: `notes` (target notes and
        the redirect warning, each already newline-terminated) in front of the header, the
        element list below ELEMENTS_MARKER."""
        header = (notes + f"{SnapshotFormat.URL_PREFIX}{url}\nTitle: {title}\n" + page_context
                  + f"\n{SnapshotFormat.ELEMENTS_MARKER}\n")
        body = '\n'.join(lines) if lines else SnapshotFormat.NO_ELEMENTS
        if hidden_frames:
            frame_lines = '\n'.join(f'  - {s}' for s in hidden_frames)
            body += (
                f"\n\n{SnapshotFormat.IFRAMES_NOTE} (their contents are NOT in the "
                "snapshot above — if the frame was just inserted, browse_snapshot() re-reads the page "
                "once it has loaded. A payment, checkout or sign-in frame belongs to this page's "
                "session, so never open its URL top-level: browse_screenshot() shows it and "
                "browse_click_at() reaches into it. Only a content frame, an embedded article or map, "
                "can be read with browse_open() on its URL):\n"
                + frame_lines
            )
        return header + body

    _BROWSE_FILL_JS = r"""
(function(el, value) {
    el.focus();
    const proto = Object.getPrototypeOf(el);
    const desc = Object.getOwnPropertyDescriptor(proto, 'value');
    const setter = desc && desc.set;
    if (setter) setter.call(el, value); else el.value = value;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    el.blur();
    return true;
})
"""

    # CDP-fill step 1: focus + clear via the native value setter (React-safe),
    # WITHOUT blurring — Input.insertText then types into the focused field.
    _BROWSE_FOCUS_CLEAR_JS = r"""
(function(el) {
    el.focus();
    const proto = Object.getPrototypeOf(el);
    const desc = Object.getOwnPropertyDescriptor(proto, 'value');
    const setter = desc && desc.set;
    if (setter) setter.call(el, '');
    else if (el.isContentEditable) el.textContent = '';
    else el.value = '';
    el.dispatchEvent(new Event('input', { bubbles: true }));
    return true;
})
"""

    # CDP-fill commit for the no-Enter case: preserve the legacy change+blur
    # semantics (date inputs, blur-committed widgets).
    _BROWSE_COMMIT_JS = r"""
(function(el) {
    el.dispatchEvent(new Event('change', { bubbles: true }));
    el.blur();
    return true;
})
"""

    _BROWSE_SELECT_JS = r"""
(function(el, value) {
    let matched = false;
    for (const opt of el.options) {
        if (opt.value === value || (opt.text || '').trim() === value) {
            el.value = opt.value;
            matched = true;
            break;
        }
    }
    if (!matched) {
        for (const opt of el.options) {
            if ((opt.text || '').trim().toLowerCase() === value.toLowerCase()) {
                el.value = opt.value;
                matched = true;
                break;
            }
        }
    }
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    return matched;
})
"""

    # Find a snapshot-tagged element in the calling context's document or any
    # same-origin child iframe. (Cross-origin child frames are separate CDP
    # targets — actions on their elements evaluate this inside that target.)
    _BROWSE_FIND_JS = r"""
(function(sel) {
    let el = document.querySelector(sel);
    if (el) return el;
    for (const f of document.querySelectorAll('iframe')) {
        try {
            const d = f.contentDocument;
            if (d) {
                const e = d.querySelector(sel);
                if (e) return e;
            }
        } catch (e) {}
    }
    return null;
})"""

    @staticmethod
    def _js_escape(value: str) -> str:
        """Escape a string so it's safe to embed inside a JS string literal."""
        return (
            value.replace('\\', '\\\\')
                 .replace("'", "\\'")
                 .replace('"', '\\"')
                 .replace('\n', '\\n')
                 .replace('\r', '\\r')
        )

    def _snapshot_frame_id(self, idx: int):
        """Owning iframe target id for a snapshot index, or None for the main page."""
        if 0 <= idx < len(self._browse_snapshot_frame_ids):
            return self._browse_snapshot_frame_ids[idx]
        return None

    async def _async_execute_action(self, idx: int, value: Optional[str],
                                    press_enter: bool = False) -> Optional[str]:
        """Execute the right action for the snapshot element at idx via in-page JS."""
        if self._tab is None:
            return "browser not available"

        sel = f'[data-floxy-snap-idx="{idx}"]'
        sel_js = sel.replace("'", "\\'")

        # Elements collected from a cross-origin iframe live in that frame's own
        # CDP target — evaluate there; main-page elements go through the tab.
        frame_id = self._snapshot_frame_id(idx)
        if frame_id is None:
            def ev(js):
                return self._evaluate(self._tab, js)
        else:
            def ev(js):
                return self._async_eval_in_frame_target(frame_id, js)

        find = f"({self._BROWSE_FIND_JS})('{sel_js}')"

        # Pull tag/type/role into Python via a single round-trip
        try:
            meta = await ev(
                f"""
                (function() {{
                    const el = {find};
                    if (!el) return null;
                    return {{
                        tag: el.tagName.toLowerCase(),
                        type: (el.getAttribute('type') || '').toLowerCase(),
                        role: (el.getAttribute('role') || '').toLowerCase(),
                        contenteditable: el.getAttribute('contenteditable'),
                        checked: el.checked === undefined ? null : !!el.checked,
                    }};
                }})();
                """,
            )
        except Exception as e:
            return f"element lookup failed: {e}"

        if not meta or not isinstance(meta, dict):
            return "element no longer in DOM"

        tag = meta.get('tag', '')
        type_ = meta.get('type', '')
        role = meta.get('role', '')
        contenteditable = meta.get('contenteditable')

        is_text_field = (value is not None and tag != 'select'
                         and not (tag == 'input' and type_ in ('checkbox', 'radio'))
                         and (tag in ('input', 'textarea')
                              or contenteditable in ('', 'true')
                              or role == 'textbox'))
        if press_enter and not is_text_field:
            desc = tag + (f'[{type_}]' if tag == 'input' and type_ else '')
            return (f"[{idx}] is a <{desc}>, not a text field — filling with \\n only works "
                    f"on text inputs. If [{idx}] opens a search box, click it plain "
                    f"(browse_act(\"{idx}\")) and fill the input in the next snapshot.")

        # Scroll into view (best-effort)
        try:
            await ev(
                f"(function(){{ const e = {find}; "
                f"if (e) e.scrollIntoView({{block:'center'}}); }})();"
            )
        except Exception:
            pass

        async def click() -> Optional[str]:
            if frame_id is None and await self._async_trusted_click(find):
                return None
            try:
                clicked = await ev(
                    f"(function(){{ const e = {find}; if (e) e.click(); return !!e; }})();"
                )
            except Exception as e:
                return f"click failed: {self._describe_error(e)}"
            if not clicked:
                if self._click_answered_consent:
                    return self.CONSENT_RELOAD_ERROR
                return "element no longer in DOM"
            return None

        # Click (no value)
        if value is None:
            return await click()

        # <select>
        if tag == 'select':
            try:
                escaped = self._js_escape(value)
                matched = await ev(
                    f"({self._BROWSE_SELECT_JS})({find}, '{escaped}');",
                )
            except Exception as e:
                return f"select failed: {e}"
            if not matched:
                return f"no option matching {value!r}"
            return None

        # checkbox / radio
        if tag == 'input' and type_ in ('checkbox', 'radio'):
            want = value.strip().lower() in ('true', '1', 'yes', 'on', 'checked')
            currently = bool(meta.get('checked'))
            if currently != want:
                if err := await click():
                    return f"toggle failed: {err}"
            return None

        # text fields / contenteditable / textbox
        if is_text_field:
            escaped = self._js_escape(value)

            async def legacy_fill() -> Optional[str]:
                try:
                    await ev(f"({self._BROWSE_FILL_JS})({find}, '{escaped}');")
                except Exception as e:
                    return f"fill failed: {e}"
                if press_enter:
                    try:
                        await self._async_dispatch_key('enter', frame_id)
                    except Exception as e:
                        return f"filled, but pressing Enter failed: {e}"
                return None

            if frame_id is not None:
                return await self._async_fill_oopif_field(idx, ev, find, value, press_enter,
                                                          frame_id, legacy_fill)

            try:
                await ev(f"({self._BROWSE_FOCUS_CLEAR_JS})({find});")
                await self._cdp_wait_for(
                    self._tab.send(cdp.input_.insert_text(text=value)), self._EVAL_TIMEOUT)
                if press_enter:
                    # Before any blur, while focus is still on the field
                    await self._async_dispatch_key('enter', frame_id)
                else:
                    await ev(f"({self._BROWSE_COMMIT_JS})({find});")
            except Exception as e:
                note(f"CDP fill on [{idx}] failed ({type(e).__name__}: {e}) — "
                     "falling back to legacy JS fill")
                return await legacy_fill()
            return None

        # Fallback — click
        return await click()

    _BROWSE_READ_VALUE_JS = r"""
(function(el) {
    if (!el) return '';
    if (el.value !== undefined && el.value !== null) return String(el.value);
    return String(el.textContent || '');
})"""

    async def _async_fill_oopif_field(self, idx: int, ev, find: str, value: str,
                                      press_enter: bool, frame_id, legacy_fill) -> Optional[str]:
        """Fill a text field that lives in a cross-origin iframe target.

        Focus + clear run inside the frame's own session; Input.insertText then
        goes to the MAIN tab — browser-level input follows focus, and the
        per-session variant proved flaky. Two guards keep a misrouted insertText
        from typing into some main-frame field: the main document's
        activeElement must be an <iframe> before typing, and the frame field must
        hold text afterwards. Either guard failing falls back to the legacy JS
        fill (native value setter + input/change events), which Stripe's card
        element is known to accept.
        """
        try:
            await ev(f"({self._BROWSE_FOCUS_CLEAR_JS})({find});")
            focus_in_frame = await self._evaluate(
                self._tab,
                "!!(document.activeElement && document.activeElement.tagName === 'IFRAME')")
            if not focus_in_frame:
                trace(f"fill [{idx}]: iframe field focused but the page's focus is not in an "
                      "iframe — using legacy JS fill")
                return await legacy_fill()
            await self._cdp_wait_for(
                self._tab.send(cdp.input_.insert_text(text=value)), self._EVAL_TIMEOUT)
            typed = await ev(f"({self._BROWSE_READ_VALUE_JS})({find});")
            if not (typed or '').strip():
                note(f"fill [{idx}]: insertText left the iframe field empty — "
                     "falling back to legacy JS fill")
                return await legacy_fill()
            trace(f"fill [{idx}]: iframe field typed via main-tab insertText ({len(typed)} chars)")
            if press_enter:
                await self._async_dispatch_key('enter', frame_id)
            else:
                await ev(f"({self._BROWSE_COMMIT_JS})({find});")
        except Exception as e:
            note(f"CDP fill on iframe field [{idx}] failed ({type(e).__name__}: {e}) — "
                 "falling back to legacy JS fill")
            return await legacy_fill()
        return None

    @staticmethod
    def _normalize_url(url: str) -> str:
        url = (url or '').strip()
        if not url:
            return url
        if url.startswith('//'):
            return 'https:' + url
        if not url.startswith(('http://', 'https://')):
            return 'https://' + url
        return url

    async def _async_navigate(self, tab: 'nodriver.Tab', url: str):
        """Send Page.navigate with the soft commit cap; log a navigation error if Chrome reports one.

        Raises asyncio.TimeoutError when the CDP response is late (page busy) —
        callers treat that as soft — and _BROWSER_DEAD_EXC on real transport death."""
        result = await self._cdp_wait_for(
            tab.send(cdp.page.navigate(url=url)), self._NAV_COMMIT_TIMEOUT)
        error_text = result[2] if isinstance(result, (tuple, list)) and len(result) > 2 else None
        if error_text:
            note(f"browse: Page.navigate to {url} reported {error_text!r} — "
                 "continuing; the snapshot will show whatever page is active")

    async def _async_browse_open(self, url: str, _retrying: bool = False) -> str:
        if err := await self._async_browse_ready():
            return err

        # A previous flow may have left popup tabs open — close them and return to
        # the base tab so the new session starts from a known single-tab state.
        await self._async_close_popup_targets()
        if self._tab is None:
            note("browse_open: no live tab after popup cleanup — reinitializing the browser")
            await self._async_initialize_driver()
            if self._tab is None:
                return await self._async_infra_error("browser could not be restarted.")

        if self._is_pdf_url(url):
            pdf_result = await self._async_fetch_pdf_bytes_via_chrome(url)
            if isinstance(pdf_result, str):
                return pdf_result
            text = await asyncio.to_thread(PdfExtractor.extract_text, pdf_result)
            if text.startswith("ERROR:"):
                return text
            self._reset_browse_session_state()
            self._browse_session_url = url
            self._browse_session_generation = self._browser_generation
            self._browse_pdf_text = text
            info(f"browse_open: PDF session for {url}, {len(text)} chars extracted")
            return (
                f"PDF document loaded from {url}.\n"
                "(PDFs have no interactive elements; browse_act/inspect/eval/screenshot are unavailable. "
                "Call browse_extract to re-read this content.)\n\n"
                + text
            )

        # Drop any prior PDF-session state when switching to a real page
        self._browse_pdf_text = None
        self._clear_network_log()

        tab = self._tab
        try:
            # Issue Page.navigate with a SOFT cap: the CDP response not arriving
            # only means the page is busy — navigation was still issued, so we
            # proceed to the readiness gate instead of stopping the load or
            # reinitializing a perfectly usable browser.
            try:
                await self._async_navigate(tab, url)
            except asyncio.TimeoutError:
                note(f"browse_open: navigation commit for {url} still pending after "
                     f"{self._NAV_COMMIT_TIMEOUT:.0f}s — continuing (soft cap)")
            except _BROWSER_DEAD_EXC as e:
                note(f"browse_open: nodriver session broken ({type(e).__name__}: {e}) — retrying")
                await self._async_initialize_driver()
                if self._tab is None:
                    return await self._async_infra_error("browser could not be restarted.")
                tab = self._tab
                try:
                    await self._async_navigate(tab, url)
                except asyncio.TimeoutError:
                    note(f"browse_open: navigation commit for {url} still pending after "
                         f"{self._NAV_COMMIT_TIMEOUT:.0f}s on retry — continuing (soft cap)")

            await self._async_wait_for_page_ready(tab)
            await self._async_wait_for_settle(tab)
            wall = await self._async_detect_bot_wall(tab, url)
            if wall is None:
                await self._async_settle_cookie_banner(tab, url)
            self._browse_session_url = url
            self._browse_session_generation = self._browser_generation
            self._browse_fail_streak = 0
            self._anchor_links_to_tab(url)
            if wall is not None and wall[0]:
                return self._take_dialog_notes() + wall[1]
            prefix = (wall[1] + "\n\n") if wall is not None else ''
            return self._take_dialog_notes() + prefix + await self._async_take_snapshot(requested_url=url)
        except _BROWSER_DEAD_EXC as e:
            note(f"browse_open: nodriver unhealthy after navigation to {url}: {e} — reinitializing")
            await self._async_initialize_driver()
            return await self._async_infra_error("browser crashed while loading the page.")
        except Exception as e:
            # Unknown failure: verify against Chrome liveness before writing it off
            # as a page error — a dead transport surfacing through an unexpected
            # exception type must still trigger recovery.
            if not _retrying and not await self._async_browser_alive():
                note(f"browse_open: opaque failure ({type(e).__name__}: {e}) with a dead browser — "
                     "restarting and retrying once")
                if await self._async_ensure_browser(force=True):
                    return await self._async_browse_open(url, _retrying=True)
                return await self._async_infra_error("browser could not be restarted.")
            trace(f"browse_open failed for {url}: {type(e).__name__}: {e}")
            return "ERROR: failed to open the page."

    # Bot walls: the document itself comes back with an error status and the page
    # embeds the vendor's challenge frame. DataDome's `t=` says which kind: 'bv'
    # (blocked visitor) is a dead end holding only a feedback form — nothing to
    # solve, and every retry lowers the IP's reputation further (Etsy, 2026-09-29:
    # three attempts in a row). 'fe' and the Cloudflare turnstile are challenges
    # the snapshot can at least show.
    _BOT_WALL_HOSTS = {'geo.captcha-delivery.com': 'DataDome',
                       'challenges.cloudflare.com': 'Cloudflare'}
    _BOT_WALL_STATUSES = frozenset({403, 429, 503})
    _BOT_WALL_FRAMES_JS = "Array.from(document.querySelectorAll('iframe')).map(f => f.src || '')"

    def _last_document_status(self, tab: 'nodriver.Tab') -> Optional[int]:
        """Status of the newest document response the network capture saw on this tab."""
        tab_id = self._network_tab_id(tab)
        for e in reversed(getattr(self, '_network_log', None) or []):
            if e['type'] == 'Document' and e['tab'] == tab_id and e['status'] is not None:
                return e['status']
        return None

    async def _async_detect_bot_wall(self, tab: 'nodriver.Tab', url: str) -> Optional[tuple[bool, str]]:
        """(hard, message) when the page is a bot wall instead of the site: hard means
        there is nothing to solve and the LLM must not retry. None for a normal page."""
        status = self._last_document_status(tab)
        if status not in self._BOT_WALL_STATUSES:
            return None
        try:
            srcs = await self._evaluate(tab, self._BOT_WALL_FRAMES_JS)
        except Exception as e:
            trace(f"bot wall: document {status} at {url} but the frame probe failed "
                  f"({type(e).__name__}: {e})")
            return None
        for src in (srcs if isinstance(srcs, list) else []):
            parsed = urlparse(str(src))
            vendor = self._BOT_WALL_HOSTS.get((parsed.hostname or '').lower())
            if not vendor:
                continue
            verdict = (parse_qs(parsed.query).get('t') or [''])[0]
            hard = vendor == 'DataDome' and verdict == 'bv'
            info(f"browse_open: {vendor} bot wall at {url} — document {status}, t={verdict or '-'}, "
                 f"{'hard block' if hard else 'challenge'} ({self._frame_label(str(src))})")
            if hard:
                return True, (
                    f"ERROR: {url} is behind a {vendor} bot wall (HTTP {status}, verdict t=bv = "
                    "'blocked visitor'). This is a hard block with no challenge to solve: the page holds "
                    "only a feedback form, and every further attempt lowers this server's reputation. "
                    "Do NOT retry browse_open on this site in this request — tell the user the site "
                    "blocks automated access.")
            return False, (
                f"NOTE: {vendor} bot challenge (HTTP {status}, t={verdict or '-'}) is shown instead of "
                f"{url}. The snapshot below is the challenge page, not the site.")
        trace(f"bot wall: document {status} at {url} without a known challenge frame — a normal error page")
        return None

    async def _with_browse_lock(self, coro):
        """Serialize browse-session commands on the shared tab. Cancelled while queued
        (watchdog): close the never-started coroutine so it doesn't warn."""
        try:
            async with self._browse_lock:
                return await coro
        finally:
            coro.close()

    async def _async_watchdog_reset(self):
        async with self._browse_lock:
            self._reset_browse_session_state()
        if not await self._async_browser_alive():
            note("watchdog reset: browser is dead — attempting restart")
            await self._async_ensure_browser()

    def _run_browse_op(self, op: str, coro, error_result: str, timeout: Optional[float] = None):
        """Run one browse command under the session lock with the watchdog applied.

        On watchdog expiry the coroutine is cancelled (see _LoopThread.submit), the
        session state is reset, and the LLM gets an error that names the timeout and
        tells it how to resume — the 2026-08-31 checkout hang surfaced as a blank
        'submit failed: ' and a 2-minute silence the model then explained with fiction.
        `timeout` overrides the watchdog for a command that legitimately waits longer
        (browse_wait).
        """
        watchdog = timeout or self._BROWSE_OP_TIMEOUT
        try:
            return self._loop_thread.submit(self._with_browse_lock(coro), timeout=watchdog)
        except BrowseWatchdogTimeout:
            note(f"{op}: cancelled by the {watchdog:.0f}s watchdog — resetting browse session state")
            try:
                self._loop_thread.submit(self._async_watchdog_reset(), timeout=15)
            except Exception as e:
                note(f"{op}: post-watchdog reset failed: {type(e).__name__}: {e}")
            return (f"ERROR: {op} was cancelled after {watchdog:.0f}s — the page or "
                    "an embedded frame stopped responding. The browse session state was reset; "
                    "call browse_open(url) for a fresh snapshot before acting again.")
        except Exception as e:
            trace(f"{op} submit failed: {type(e).__name__}: {e}")
            return error_result

    def browse_open(self, url: str) -> str:
        """Open a URL in the browse session and return an interactive-element snapshot."""
        url = self._normalize_url(self._resolve_page_link(url))
        if not url:
            return "ERROR: no URL provided to browse_open."
        info(f"browse_open: {url}")
        return self._run_browse_op(f"browse_open({url})", self._async_browse_open(url),
                                   "ERROR: failed to open the page.")

    def _pdf_session_error(self, capability: str) -> Optional[str]:
        if self._browse_pdf_text is None:
            return None
        return f"ERROR: current session is a PDF — {capability}. Use browse_extract for the text."

    def _reset_browse_session_state(self) -> None:
        if self._popup_parents:
            base = self._popup_parents[0]
            # Sync context — no liveness fetch possible. Best effort: restore the
            # base tab only if the last-known live set contained it; the async
            # popup paths above do the verified restores.
            if str(base.target.target_id) in self._known_page_targets:
                self._tab = base
            else:
                trace("reset: base tab absent from last-known targets — keeping the current tab")
            self._popup_parents = []
        self._known_page_targets = set()
        self._pending_popup_targets = {}
        self._pending_target_notes = []
        self._dialog_notes = []
        self._browse_snapshot_frame_ids = []
        self._browse_snapshot_labels = []
        self._browse_frame_sessions = {}
        self._blank_frame_target_ids = frozenset()
        self._oopif_settle_giveups = frozenset()
        self._last_screenshot_bytes = None
        self._last_screenshot_path = None

    # Split on ';' only where the next segment starts a new op (an index followed
    # by '=', ';' or end) — a ';' inside a fill value stays part of the value.
    _ACT_BATCH_SPLIT_RE = re.compile(r';\s*(?=\d+\s*(?:=|;|$))')

    @classmethod
    def _split_act_batch(cls, spec: str) -> list[str]:
        return [p.strip() for p in cls._ACT_BATCH_SPLIT_RE.split(spec) if p.strip()]

    @staticmethod
    def _fmt_op_list(ops: list) -> str:
        return ', '.join(f"'{op_spec}'" for op_spec, _, _, _ in ops) or '(none)'

    def _current_tab_url(self) -> str:
        try:
            return self._tab.target.url or ''
        except Exception:
            return ''

    def _anchor_links_to_tab(self, fallback: Optional[str] = None) -> None:
        """Make the tab's page the one a root-relative link is resolved against."""
        tab_url = self._current_tab_url()
        base = tab_url if tab_url.startswith(('http://', 'https://')) else fallback
        if base and base != getattr(self, '_link_base_url', None):
            trace(f"Links now resolve against {base}")
            self._link_base_url = base

    async def _async_page_probe(self, tab: 'nodriver.Tab') -> Optional[dict]:
        """One read of the page's state: readyState, the text signature and whether a
        progress indicator is showing. None when the page cannot be asked."""
        try:
            probe = await self._cdp_wait_for(self._evaluate(tab, self._PAGE_PROBE_JS),
                                             self._PAGE_READY_EVAL_TIMEOUT)
        except Exception as e:
            trace(f"page probe failed ({type(e).__name__}: {e})")
            return None
        return probe if isinstance(probe, dict) and 'sig' in probe else None

    async def _async_action_mark(self) -> dict:
        """The page as it is right before a click/key press: its text signature (None when
        the page cannot be asked), for _async_snapshot_after_action to tell a reaction from
        silence, and the moment, so the requests the action sends can be told from older ones."""
        mark = {'sig': None, 'net_ts': time.time()}
        if self._tab is not None:
            probe = await self._async_page_probe(self._tab)
            if probe:
                mark['sig'] = probe['sig']
        return mark

    async def _async_wait_for_page_change(self, mark: dict) -> tuple[str, Optional[dict]]:
        """After a click/Enter: wait up to _ACTION_CHANGE_TIMEOUT for the page to start
        reacting — the text signature moved (URL included) or a navigation is loading — so
        what follows measures the new page rather than the old one sitting quietly while
        the request is in flight. Soft: every failure just returns. Returns what was seen
        and the last probe."""
        started = time.time()
        deadline = started + self._ACTION_CHANGE_TIMEOUT
        probe = None
        while True:
            tab = self._tab
            if tab is None:
                return 'no-tab', None
            probe = await self._async_page_probe(tab)
            if probe is None:
                return 'probe-failed', None
            if probe.get('state') == 'loading':
                trace(f"page-change: navigation in progress after {time.time() - started:.2f}s")
                return 'navigating', probe
            if mark.get('sig') is None or probe['sig'] != mark['sig']:
                trace(f"page-change: text changed after {time.time() - started:.2f}s")
                return 'changed', probe
            if time.time() >= deadline:
                trace(f"page-change: nothing changed within {self._ACTION_CHANGE_TIMEOUT}s of the action")
                return 'timeout', probe
            await asyncio.sleep(self._ACTION_CHANGE_POLL)

    async def _async_wait_for_settle(self, tab: 'nodriver.Tab', sig: Optional[str] = None) -> str:
        """Wait for the page to stop working on itself: no visible progress indicator and
        the text signature unchanged between two probes (`sig` is the last one seen, so a
        page that has not moved since costs one probe). Soft cap _SETTLE_TIMEOUT — a
        spinner that waits for the user (a 2FA prompt) never settles."""
        started = time.time()
        deadline = started + self._SETTLE_TIMEOUT
        while True:
            probe = await self._async_page_probe(tab)
            if probe is None:
                return 'probe-failed'
            if not probe.get('busy') and sig is not None and probe['sig'] == sig:
                fine(f"settle: page steady after {time.time() - started:.2f}s")
                return 'settled'
            if time.time() >= deadline:
                trace(f"settle: page still {'busy' if probe.get('busy') else 'changing'} after "
                      f"{self._SETTLE_TIMEOUT}s — proceeding as-is (soft cap)")
                return 'timeout'
            sig = probe['sig']
            await asyncio.sleep(self._SETTLE_POLL)

    async def _async_snapshot_after_action(self, mark: Optional[dict], what: str,
                                           verdict: bool = True) -> str:
        """The common tail of browse_act / browse_key / browse_click_at: let the page react
        to the action, wait for it to settle, snapshot. With a mark (the action could change
        the page: a click, Enter, a key) a page whose text signature still equals the one
        before the action gets one more wait and a re-read, and stays flagged as unchanged
        if it still is — the LLM must not act on a page that only looks like the one it
        already saw. `verdict` False keeps the waits but skips that re-read and flag, for an
        action whose effect the signature cannot see (a click inside an iframe)."""
        probe = None
        if mark is not None:
            _, probe = await self._async_wait_for_page_change(mark)
        await asyncio.sleep(self._POPUP_SETTLE_S)
        await self._async_sync_popup_targets()
        if self._tab is not None:
            await self._async_wait_for_page_ready(self._tab)
            if mark is not None:
                await self._async_wait_for_settle(self._tab, probe['sig'] if probe else None)
        snapshot = await self._async_take_snapshot()
        before = mark.get('sig') if mark else None
        if before is None or self._tab is None or not verdict:
            return snapshot
        probe = await self._async_page_probe(self._tab)
        if probe is None or probe['sig'] != before:
            return snapshot
        info(f"{what}: page unchanged by the action — "
             f"waiting {self._ACTION_UNCHANGED_RETRY_S}s and re-reading")
        await asyncio.sleep(self._ACTION_UNCHANGED_RETRY_S)
        await self._async_sync_popup_targets()
        if self._tab is not None:
            await self._async_wait_for_page_ready(self._tab)
            await self._async_wait_for_settle(self._tab, probe['sig'])
        snapshot = await self._async_take_snapshot()
        probe = await self._async_page_probe(self._tab) if self._tab is not None else None
        if probe is None or probe['sig'] != before:
            return snapshot
        note(f"{what}: page unchanged after the action")
        head = (f"NOTE: the page looks exactly as it did before {what} — the action had no visible "
                "effect, or the page is still working on it. If a change is expected, call "
                "browse_snapshot() to re-read it.")
        # A server that refuses the submit with nothing drawn on the page (the 2026-10-22 Tesla
        # checkout: Nästa did nothing, the 400 behind it sat in the capture) — say so here,
        # where the LLM looks, rather than leave it to find browse_network five rounds later.
        failures = self._network_failures_since(mark.get('net_ts'))
        if failures:
            info(f"{what}: {len(failures)} request(s) sent since the action failed — listed with the snapshot")
            head += ("\nRequests sent since the action that the server refused or that failed, the likely "
                     "reason nothing moved (browse_network has the rest):\n"
                     + "\n".join(self._format_network_entry(i, e) for i, e in failures))
        else:
            fine(f"{what}: no failed request since the action to explain the unchanged page")
        return head + "\n\n" + snapshot

    async def _async_browse_act(self, spec: str) -> str:
        if not self._browse_session_url:
            return "ERROR: no browse session. Call browse_open(url) first."
        if err := self._pdf_session_error("no interactive elements"):
            return err

        # A real trailing newline is also a press-Enter marker — check before
        # strip() eats it (defensive; the parser normally delivers a literal \n).
        real_newline_marker = (spec or '').rstrip(' \t').endswith('\n')

        spec = (spec or '').strip()
        if not spec:
            return "ERROR: browse_act needs an argument like '3', '3=value' or '3=a; 4=b; 5'."

        op_specs = self._split_act_batch(spec)
        ops = []  # (op_spec, idx, value, press_enter)
        for n, op_spec in enumerate(op_specs):
            press_enter = False
            if '=' in op_spec:
                idx_str, _, value = op_spec.partition('=')
                value = value.strip()
                if value.endswith('\\n'):
                    # CommandParser.split_parameters doesn't resolve \n escapes, so the
                    # LLM's "...\n" arrives as a literal backslash-n — the primary marker.
                    press_enter = True
                    value = value[:-2].rstrip()
                elif real_newline_marker and n == len(op_specs) - 1:
                    press_enter = True
            else:
                idx_str, value = op_spec, None
            idx_str = idx_str.strip()

            try:
                idx = int(idx_str)
            except ValueError:
                return f"ERROR: '{idx_str}' is not a valid index."
            ops.append((op_spec, idx, value, press_enter))

        if gate := await self._async_browse_ready_midsession():
            return gate
        if not self._browse_snapshot_frame_ids:
            return "ERROR: no current snapshot. Call browse_open(url) again."

        # Every index in the batch refers to the CURRENT snapshot — validate all
        # of them up front so a bad one aborts before anything runs.
        for _, idx, _, _ in ops:
            if idx < 0 or idx >= len(self._browse_snapshot_frame_ids):
                note(f"browse_act: index {idx} out of range (snapshot has {len(self._browse_snapshot_frame_ids)})")
                return (f"ERROR: index {idx} is out of range. Take a fresh snapshot first."
                        + (" No batched action was executed.\n\n" if len(ops) > 1 else "\n\n")
                        + await self._async_take_snapshot())

        if len(ops) > 1:
            info(f"browse_act: batch of {len(ops)} actions")
        start_url = self._current_tab_url()
        warning = ''
        mark = None
        for n, (op_spec, idx, value, press_enter) in enumerate(ops):
            if value is None:
                info(f"browse_act: click [{idx}]")
            else:
                info(f"browse_act: fill [{idx}] = {value!r}"
                     + (" + press Enter" if press_enter else ""))

            # Only a click or Enter can make the page move on — a fill just edits a
            # field, and the value it leaves shows in the snapshot anyway.
            mark = await self._async_action_mark() if (value is None or press_enter) else None
            err = await self._async_execute_action(idx, value, press_enter)
            if err:
                note(f"browse_act failed at [{idx}]: {err}")
                msg = f"ERROR: action on [{idx}] failed: {err}"
                if len(ops) > 1:
                    msg += (f"\nBatch stopped: executed {self._fmt_op_list(ops[:n])} before the "
                            f"failure; skipped {self._fmt_op_list(ops[n + 1:])}.")
                if err == self.CONSENT_RELOAD_ERROR:
                    # the page is reloading after the consent: snapshot it once it is back
                    snapshot = await self._async_snapshot_after_action(mark, f"browse_act('{op_spec}')",
                                                                       verdict=False)
                else:
                    snapshot = await self._async_take_snapshot()
                return msg + "\n\n" + snapshot

            # A mid-batch navigation invalidates every remaining index — stop and
            # let the LLM re-aim against the fresh snapshot below.
            if n < len(ops) - 1:
                url_now = self._current_tab_url()
                if url_now and start_url and self._urls_differ(start_url, url_now):
                    note(f"browse_act: [{idx}] navigated to {url_now} — "
                         f"stopping batch, {len(ops) - n - 1} action(s) skipped")
                    warning = (f"WARNING: action '{op_spec}' navigated to a new page, so the rest of "
                               f"the batch was skipped — those indices no longer apply. Executed "
                               f"{self._fmt_op_list(ops[:n + 1])}; skipped {self._fmt_op_list(ops[n + 1:])}. "
                               f"Re-aim using the fresh snapshot below.")
                    break

        snapshot = await self._async_snapshot_after_action(mark, f"browse_act('{ops[-1][0]}')")
        return self._take_dialog_notes() + (warning + "\n\n" + snapshot if warning else snapshot)

    def browse_act(self, spec: str) -> str:
        """Click or fill elements by snapshot index. spec: "<index>", "<index>=<value>",
        or a ';'-separated batch of those executed in order against the current snapshot."""
        return self._run_browse_op(f"browse_act({spec})", self._async_browse_act(spec),
                                   "ERROR: action failed.")

    async def _async_browse_key(self, spec: str) -> str:
        if not self._browse_session_url:
            return "ERROR: no browse session. Call browse_open(url) first."
        if err := self._pdf_session_error("no keyboard input"):
            return err

        spec = (spec or '').strip()
        if not spec:
            return "ERROR: browse_key needs a key like 'Enter' or '3,ArrowDown'."

        idx = None
        if ',' in spec:
            idx_str, _, key_name = spec.partition(',')
            key_name = key_name.strip()
            try:
                idx = int(idx_str.strip())
            except ValueError:
                return f"ERROR: '{idx_str.strip()}' is not a valid index."
        else:
            key_name = spec

        if self._resolve_key(key_name) is None:
            supported = ', '.join(d[0] for d in self._KEY_DEFS.values())
            return f"ERROR: unsupported key {key_name!r}. Supported keys: {supported}."

        if gate := await self._async_browse_ready_midsession():
            return gate
        if self._tab is None:
            return "ERROR: browser not available."

        frame_id = None
        if idx is not None:
            if not self._browse_snapshot_frame_ids:
                return "ERROR: no current snapshot. Call browse_open(url) again."
            if idx < 0 or idx >= len(self._browse_snapshot_frame_ids):
                note(f"browse_key: index {idx} out of range (snapshot has "
                     f"{len(self._browse_snapshot_frame_ids)})")
                return (f"ERROR: index {idx} is out of range. Take a fresh snapshot first.\n\n"
                        + await self._async_take_snapshot())
            frame_id = self._snapshot_frame_id(idx)
            sel_js = f'[data-floxy-snap-idx="{idx}"]'.replace("'", "\\'")
            focus_js = (f"(function(){{ const e = ({self._BROWSE_FIND_JS})('{sel_js}'); "
                        f"if (e) e.focus(); return !!e; }})();")
            try:
                if frame_id is None:
                    focused = await self._evaluate(self._tab, focus_js)
                else:
                    focused = await self._async_eval_in_frame_target(frame_id, focus_js)
            except Exception as e:
                note(f"browse_key: focusing [{idx}] failed: {e}")
                return (f"ERROR: could not focus element [{idx}]: {e}\n\n"
                        + await self._async_take_snapshot())
            if not focused:
                note(f"browse_key: element [{idx}] no longer in DOM")
                return (f"ERROR: element [{idx}] is no longer in the DOM. "
                        "Take a fresh snapshot first.\n\n"
                        + await self._async_take_snapshot())

        info(f"browse_key: {key_name}" + (f" on [{idx}]" if idx is not None else " on focused element"))
        mark = await self._async_action_mark()
        try:
            await self._async_dispatch_key(key_name, frame_id)
        except Exception as e:
            note(f"browse_key: dispatch of {key_name!r} failed: {self._describe_error(e)}")
            return f"ERROR: key press failed: {self._describe_error(e)}"

        # Enter typically triggers navigation/search — same tail as browse_act.
        return self._take_dialog_notes() + await self._async_snapshot_after_action(
            mark, f"browse_key('{spec}')")

    def browse_key(self, spec: str) -> str:
        """Press a real keyboard key in the browse session. spec: "<key>" or "<index>,<key>"."""
        return self._run_browse_op(f"browse_key({spec})", self._async_browse_key(spec),
                                   "ERROR: key press failed.")

    async def _async_browse_snapshot(self) -> str:
        if not self._browse_session_url:
            return "ERROR: no browse session. Call browse_open(url) first."
        if err := self._pdf_session_error("no interactive elements"):
            return err
        if gate := await self._async_browse_ready_midsession():
            return gate
        await self._async_sync_popup_targets()
        if self._tab is not None:
            await self._async_wait_for_page_ready(self._tab)
            await self._async_wait_for_settle(self._tab)
        return self._take_dialog_notes() + await self._async_take_snapshot()

    def browse_snapshot(self, _params=None) -> str:
        """Re-read the current page without acting: a fresh snapshot of the page as
        it is now, including iframes that were still loading when the last
        snapshot was taken."""
        info("browse_snapshot")
        return self._run_browse_op("browse_snapshot", self._async_browse_snapshot(),
                                   "ERROR: snapshot failed.")

    _WAIT_SPEC_RE = re.compile(r'^(.*?)(?:\s*,\s*(\d+(?:\.\d+)?)\s*)?$', re.S)

    @classmethod
    def _parse_wait_spec(cls, spec: str) -> tuple[str, float]:
        """'pattern' or 'pattern, seconds' → (pattern, timeout). The pattern keeps any
        comma of its own; only a trailing numeric segment is the timeout."""
        m = cls._WAIT_SPEC_RE.match((spec or '').strip())
        pattern = (m.group(1) if m else spec or '').strip().strip('"\'')
        timeout = float(m.group(2)) if m and m.group(2) else cls._WAIT_DEFAULT_S
        return pattern, timeout

    async def _async_browse_wait(self, pattern: str, timeout_s: float) -> str:
        if not self._browse_session_url:
            return "ERROR: no browse session. Call browse_open(url) first."
        if err := self._pdf_session_error("nothing to wait for"):
            return err
        if not pattern:
            return "ERROR: browse_wait needs a text or URL pattern (regex) to wait for."
        try:
            rx = re.compile(pattern, re.I | re.S)
        except re.error as e:
            trace(f"browse_wait: {pattern!r} is not a valid regex ({e}) — matching it literally")
            rx = re.compile(re.escape(pattern), re.I)
        timeout_s = max(1.0, min(timeout_s, self._WAIT_MAX_S))
        if gate := await self._async_browse_ready_midsession():
            return gate

        info(f"browse_wait: for /{pattern}/ up to {timeout_s:.0f}s")
        started = time.time()
        deadline = started + timeout_s
        polls = 0
        last_url = ''
        while True:
            # The handoff being waited for is often a popup closing or opening —
            # follow it so the probe reads the page the user would be looking at.
            await self._async_sync_popup_targets()
            tab = self._tab
            if tab is None:
                return "ERROR: browser not available."
            state = None
            try:
                state = await self._cdp_wait_for(self._evaluate(tab, self._WAIT_PROBE_JS),
                                                 self._PAGE_READY_EVAL_TIMEOUT)
            except Exception as e:
                trace(f"browse_wait: page probe failed ({self._describe_error(e)}) — retrying")
            polls += 1
            if isinstance(state, dict):
                url = state.get('url') or ''
                if url != last_url:
                    trace(f"browse_wait: page is at {url}")
                    last_url = url
                m = (rx.search(url) or rx.search(state.get('title') or '')
                     or rx.search(state.get('text') or ''))
                if m:
                    elapsed = time.time() - started
                    info(f"browse_wait: matched {m.group(0)[:80]!r} after {elapsed:.1f}s "
                         f"({polls} polls) on {url}")
                    await self._async_wait_for_page_ready(tab)
                    await self._async_wait_for_settle(tab)
                    snapshot = await self._async_take_snapshot()
                    return (self._take_dialog_notes()
                            + f"MATCHED {m.group(0)[:80]!r} after {elapsed:.1f}s.\n\n" + snapshot)
            if time.time() >= deadline:
                break
            await asyncio.sleep(self._WAIT_POLL_S)

        note(f"browse_wait: no match for /{pattern}/ within {timeout_s:.0f}s ({polls} polls) — "
             f"page still at {last_url}")
        snapshot = await self._async_take_snapshot()
        return (self._take_dialog_notes()
                + f"TIMEOUT: nothing matching {pattern!r} appeared within {timeout_s:.0f}s — below is "
                  "the page as it is now. browse_network() shows what the page has been polling "
                  "meanwhile, which usually says why.\n\n" + snapshot)

    def browse_wait(self, spec: str) -> str:
        """Wait for a text/URL pattern to appear in the browse session. spec: "<regex>" or
        "<regex>, <seconds>" (default 30 s, at most 60 s). Returns a fresh snapshot either way."""
        pattern, timeout_s = self._parse_wait_spec(spec)
        timeout_s = max(1.0, min(timeout_s, self._WAIT_MAX_S))
        # The wait overshoots its deadline by up to a poll, then page-ready, settle and the
        # snapshot follow: the watchdog leaves them the room they may take on a slow page.
        return self._run_browse_op(f"browse_wait({pattern})", self._async_browse_wait(pattern, timeout_s),
                                   "ERROR: wait failed.", timeout=timeout_s + self._BROWSE_OP_TIMEOUT / 2)

    _CLICK_AT_SPEC_RE = re.compile(r'^\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*$')

    @classmethod
    def _parse_click_at_spec(cls, spec: str) -> Optional[tuple[float, float]]:
        m = cls._CLICK_AT_SPEC_RE.match(spec or '')
        return (float(m.group(1)), float(m.group(2))) if m else None

    async def _async_browse_click_at(self, spec: str) -> str:
        if not self._browse_session_url:
            return "ERROR: no browse session. Call browse_open(url) first."
        if err := self._pdf_session_error("no interactive elements"):
            return err
        point = self._parse_click_at_spec(spec)
        if point is None:
            return ("ERROR: browse_click_at needs viewport coordinates like '412, 583' "
                    "(CSS pixels, as getBoundingClientRect reports).")
        x, y = point
        if gate := await self._async_browse_ready_midsession():
            return gate
        if self._tab is None:
            return "ERROR: browser not available."

        # What the main frame has under the point — an <iframe> means the click
        # lands inside a cross-origin frame no snapshot could list.
        hit = None
        try:
            hit = await self._evaluate(
                self._tab,
                f"(function(){{ const e = document.elementFromPoint({x}, {y}); if (!e) return null;"
                " let d = e.tagName.toLowerCase(); if (e.id) d += '#' + e.id;"
                " const src = e.getAttribute && e.getAttribute('src');"
                " if (src) d += ' src=' + src.split('?')[0].slice(0, 100);"
                " const t = (e.innerText || e.value || '').trim().slice(0, 60);"
                " if (t) d += ' \"' + t + '\"'; return d; })()")
        except Exception as e:
            trace(f"browse_click_at: elementFromPoint failed: {self._describe_error(e)}")
        if hit is None:
            note(f"browse_click_at: nothing under ({x:g}, {y:g}) — outside the viewport?")
            return (f"ERROR: nothing is rendered at ({x:g}, {y:g}) — coordinates are CSS pixels "
                    "relative to the viewport, so scroll the target into view first "
                    "(browse_eval: el.scrollIntoView()) and re-read its getBoundingClientRect().")

        info(f"browse_click_at: ({x:g}, {y:g}) on {hit}")
        # Inside an iframe the main document's text signature cannot see what the
        # click did, so the unchanged-page verdict would be noise: the page still gets
        # its time to react (a 3-D Secure submit drives the parent), the verdict is
        # skipped and the reply says why.
        in_frame = hit.startswith('iframe')
        mark = await self._async_action_mark()
        try:
            await self._async_dispatch_click_at(x, y)
        except Exception as e:
            return f"ERROR: click at ({x:g}, {y:g}) failed: {self._describe_error(e)}"

        snapshot = await self._async_snapshot_after_action(mark, f"the click at ({x:g}, {y:g})",
                                                           verdict=not in_frame)
        frame_note = ''
        if in_frame:
            frame_note = ("NOTE: the click landed inside an iframe, and what the frame did is not "
                          "visible in this snapshot of the page around it — browse_screenshot() shows "
                          "it; browse_snapshot() lists the frame's fields when it can be read.\n\n")
        return (self._take_dialog_notes()
                + f"Clicked at ({x:g}, {y:g}) — hit <{hit}>.\n\n"
                + frame_note + snapshot)

    # The point a real mouse click on the element lands on: the centre of its box when
    # that lies in the viewport and nothing else is drawn over it (a cookie veil, a modal),
    # else the reason it is not clickable and, when something covers it, what that is.
    # Only main-document elements: a frame's rect is in the frame's own coordinates,
    # not the viewport's the click is dispatched in.
    _CLICK_POINT_JS = """
(function(e) {
    const describe = function(n) {
        if (!n || !n.tagName) return String(n);
        let s = n.tagName.toLowerCase();
        if (n.id) s += '#' + n.id;
        if (typeof n.className === 'string' && n.className.trim())
            s += '.' + n.className.trim().split(/\\s+/).slice(0, 3).join('.');
        return s;
    };
    if (!e) return {reason: 'element gone'};
    if (e.ownerDocument !== document) return {reason: 'inside an iframe'};
    const r = e.getBoundingClientRect();
    if (!r.width || !r.height) return {reason: 'no box (display:none or empty)'};
    const x = r.left + r.width / 2, y = r.top + r.height / 2;
    if (x < 0 || y < 0 || x > window.innerWidth || y > window.innerHeight)
        return {reason: 'centre outside the viewport', x: Math.round(x), y: Math.round(y),
                viewport: window.innerWidth + 'x' + window.innerHeight};
    const hit = document.elementFromPoint(x, y);
    if (!hit) return {reason: 'nothing at the point'};
    if (!(e.contains(hit) || hit.contains(e)))
        return {reason: 'covered', hit: describe(hit), target: describe(e),
                shadow: !!hit.shadowRoot, inShadow: e.getRootNode() !== document};
    return {x: x, y: y};
})"""

    async def _async_trusted_click(self, find: str) -> bool:
        """Click the element `find` evaluates to with a real mouse event (isTrusted,
        mousemove before the press — what a bot score looks for in a form's submit),
        when its centre is visible on the main page. False means the caller should
        fall back to the JS click; the reason is in the log."""
        self._click_answered_consent = False
        point = await self._async_click_point(find)
        if isinstance(point, dict) and point.get('reason') == 'covered':
            # A consent wall that mounted after browse_open's pass (fedex.com's
            # Usercentrics layer, #82) sits over the whole page: answer it and look again.
            if await self._async_consent_pass_for_click(point):
                point = await self._async_click_point_once_uncovered(find)
        if not isinstance(point, dict) or 'reason' in point:
            why = point.get('reason') if isinstance(point, dict) else f"unexpected result {point!r}"
            detail = {k: v for k, v in point.items() if k != 'reason'} if isinstance(point, dict) else {}
            trace(f"trusted click: {why}{' ' + str(detail) if detail else ''} — JS click")
            return False
        try:
            await self._async_dispatch_click_at(point['x'], point['y'])
        except Exception as e:
            trace(f"trusted click at ({point['x']:g}, {point['y']:g}) failed "
                  f"({self._describe_error(e)}) — JS click")
            return False
        trace(f"trusted click at ({point['x']:g}, {point['y']:g})")
        return True

    async def _async_click_point(self, find: str):
        """The click-point verdict of _CLICK_POINT_JS for `find`, or None when the
        lookup itself failed (logged)."""
        try:
            return await self._evaluate(self._tab, f"({self._CLICK_POINT_JS})({find});")
        except Exception as e:
            trace(f"trusted click: click point lookup failed ({self._describe_error(e)}) — JS click")
            return None

    _UNCOVER_WAIT = 1.5
    _UNCOVER_STEP = 0.1
    # Set by the consent pass a trusted click runs; a click that then finds its element
    # gone reports the page's reaction to the consent rather than a stale index.
    _click_answered_consent = False
    CONSENT_RELOAD_ERROR = ("a consent banner covering the page was answered first and the page "
                            "re-rendered, so the index no longer applies — re-aim with the fresh "
                            "snapshot below")

    async def _async_click_point_once_uncovered(self, find: str):
        """The click point looked up again after a consent pass, waiting out the
        banner's closing (Usercentrics still covers the page 6 ms after the click)
        up to _UNCOVER_WAIT; the last verdict when it never clears."""
        deadline = time.monotonic() + self._UNCOVER_WAIT
        while True:
            await asyncio.sleep(self._UNCOVER_STEP)
            point = await self._async_click_point(find)
            if not (isinstance(point, dict) and point.get('reason') == 'covered'):
                return point
            if time.monotonic() >= deadline:
                trace(f"trusted click: still covered by {point.get('hit')} {self._UNCOVER_WAIT}s "
                      f"after the consent pass")
                return point

    async def _async_consent_pass_for_click(self, point: dict) -> bool:
        """One consent pass because `point` says something covers the click target;
        True when it answered a banner, so the point is worth looking up again."""
        try:
            url = await self._evaluate(self._tab, "location.href")
        except Exception:
            url = ''
        url = url if isinstance(url, str) else ''
        if self._is_auth_url(url):
            trace(f"trusted click: covered by {point.get('hit')} on an auth URL — no consent pass")
            return False
        result = await self._async_remove_cookies_banner(self._tab, url)
        clicked, removed = result.get('clicked', 0), result.get('removed', 0)
        if clicked or removed:
            info(f"trusted click: covered by {point.get('hit')} — consent pass clicked={clicked} "
                 f"removed={removed}, looking up the click point again")
            self._click_answered_consent = True
            return True
        trace(f"trusted click: covered by {point.get('hit')} — the consent pass matched nothing")
        return False

    async def _async_dispatch_click_at(self, x: float, y: float) -> None:
        """One real left click at viewport coordinates via Input.dispatchMouseEvent on
        the main tab: the browser routes it into whatever frame is drawn there, so
        it reaches fields inside cross-origin iframes as well."""
        left = cdp.input_.MouseButton.LEFT
        for type_, button in (('mouseMoved', None), ('mousePressed', left), ('mouseReleased', left)):
            kwargs = dict(type_=type_, x=x, y=y)
            if button is not None:
                kwargs.update(button=button, click_count=1)
            await self._cdp_wait_for(
                self._tab.send(cdp.input_.dispatch_mouse_event(**kwargs)), self._EVAL_TIMEOUT)

    def browse_click_at(self, spec: str) -> str:
        """Real mouse click at viewport coordinates — the fallback for elements no
        snapshot lists (fields inside a cross-origin iframe). Returns a fresh snapshot."""
        return self._run_browse_op(f"browse_click_at({spec})", self._async_browse_click_at(spec),
                                   "ERROR: click failed.")
    _BROWSE_INSPECT_JS = r"""
(function(sel, attr) {
    const findEl = function() {
        let el = document.querySelector(sel);
        if (el) return el;
        for (const f of document.querySelectorAll('iframe')) {
            try {
                const d = f.contentDocument;
                if (d) {
                    const e = d.querySelector(sel);
                    if (e) return e;
                }
            } catch (e) {}
        }
        return null;
    };
    const el = findEl();
    if (!el) return { error: 'not_found' };

    const TRUNC_OUTER = 2048;
    const TRUNC_TEXT = 500;
    const TRUNC_VAL = 200;

    const truncate = function(s, max) {
        if (s == null) return '';
        s = String(s);
        if (s.length <= max) return s;
        return s.slice(0, max) + '… [truncated, ' + s.length + ' chars total]';
    };

    if (attr) {
        // Targeted mode — return raw, untruncated value.
        if (attr === 'outerHTML') return { value: el.outerHTML || '' };
        if (attr === 'innerHTML') return { value: el.innerHTML || '' };
        if (attr === 'textContent') return { value: el.textContent || '' };
        if (attr === 'value') {
            if (el.value !== undefined) return { value: String(el.value) };
            if (el.isContentEditable) return { value: el.innerText || '' };
            return { value: '', missing: true };
        }
        if (el.hasAttribute(attr)) return { value: el.getAttribute(attr) || '' };
        return { value: '', missing: true };
    }

    // Discovery mode — dump all attributes plus a few derived properties.
    const out = {
        tag: el.tagName.toLowerCase(),
        attributes: {},
        outerHTML: truncate(el.outerHTML || '', TRUNC_OUTER),
        textContent: truncate((el.textContent || '').trim(), TRUNC_TEXT),
    };
    for (const a of el.attributes) {
        out.attributes[a.name] = truncate(a.value, TRUNC_VAL);
    }
    if (el.value !== undefined) {
        out.value = truncate(String(el.value), TRUNC_VAL);
    }
    return out;
})
"""

    @staticmethod
    def _format_inspect_result(idx: int, attr: Optional[str], result) -> str:
        if not isinstance(result, dict):
            return f"ERROR: inspect on [{idx}] returned no data."
        if result.get('error') == 'not_found':
            return f"ERROR: element [{idx}] no longer in DOM. Take a fresh snapshot."

        if attr:
            if result.get('missing'):
                return f"[{idx}] attribute {attr!r}: (not set on this element)"
            return result.get('value', '')

        lines = [f"[{idx}] <{result.get('tag', '?')}>"]
        attrs = result.get('attributes') or {}
        if attrs:
            lines.append("attributes:")
            for name, value in attrs.items():
                lines.append(f"  {name}={value!r}")
        else:
            lines.append("attributes: (none)")
        if 'value' in result:
            lines.append(f"value={result['value']!r}")
        text = result.get('textContent') or ''
        if text:
            lines.append(f"textContent={text!r}")
        outer = result.get('outerHTML') or ''
        if outer:
            lines.append(f"outerHTML={outer}")
        lines.append(
            "(values truncated; use browse_inspect(idx, <name>) to read a single attribute in full. "
            "JS handlers attached via addEventListener are not visible.)"
        )
        return '\n'.join(lines)

    async def _async_browse_inspect(self, spec: str) -> str:
        if not self._browse_session_url:
            return "ERROR: no browse session. Call browse_open(url) first."
        if err := self._pdf_session_error("no DOM elements to inspect"):
            return err

        spec = (spec or '').strip()
        if not spec:
            return "ERROR: browse_inspect needs an argument like '3' or '3,href'."

        if ',' in spec:
            idx_str, _, attr = spec.partition(',')
            attr = attr.strip()
        else:
            idx_str, attr = spec, ''
        idx_str = idx_str.strip()

        try:
            idx = int(idx_str)
        except ValueError:
            return f"ERROR: '{idx_str}' is not a valid index."

        if gate := await self._async_browse_ready_midsession():
            return gate
        if not self._browse_snapshot_frame_ids:
            return "ERROR: no current snapshot. Call browse_open(url) again."

        if idx < 0 or idx >= len(self._browse_snapshot_frame_ids):
            note(f"browse_inspect: index {idx} out of range (snapshot has {len(self._browse_snapshot_frame_ids)})")
            return f"ERROR: index {idx} is out of range. Take a fresh snapshot first."

        if attr:
            info(f"browse_inspect: [{idx}] attribute {attr!r}")
        else:
            info(f"browse_inspect: [{idx}] full dump")

        sel = f'[data-floxy-snap-idx="{idx}"]'
        sel_js = self._js_escape(sel)
        attr_js = self._js_escape(attr) if attr else ''
        js = f"({self._BROWSE_INSPECT_JS})('{sel_js}', '{attr_js}');"
        frame_id = self._snapshot_frame_id(idx)
        try:
            if frame_id is None:
                result = await self._evaluate(self._tab, js)
            else:
                result = await self._async_eval_in_frame_target(frame_id, js)
        except Exception as e:
            trace(f"browse_inspect [{idx}] failed: {e}")
            return f"ERROR: inspect on [{idx}] failed: {e}"

        return self._format_inspect_result(idx, attr or None, result)

    def browse_inspect(self, spec: str) -> str:
        """Read attributes/HTML/text of a snapshot element. spec: "<index>" or "<index>,<attr>".

        Discovery mode (no attr): dumps all attributes plus outerHTML/textContent (truncated).
        Targeted mode (with attr): returns the full untruncated value of that attribute.
        Special attr names: outerHTML, innerHTML, textContent, value."""
        return self._run_browse_op(f"browse_inspect({spec})", self._async_browse_inspect(spec),
                                   "ERROR: inspect failed.")

    async def _async_browse_extract(self) -> str:
        if not self._browse_session_url:
            return "ERROR: no browse session. Call browse_open(url) first."

        if self._browse_pdf_text is not None:
            info(f"browse_extract: returning cached PDF text ({len(self._browse_pdf_text)} chars)")
            return self._browse_pdf_text

        if gate := await self._async_browse_ready_midsession():
            return gate

        url = self._browse_session_url
        try:
            url = self._tab.target.url or url
        except Exception:
            pass

        try:
            page_source = await self._tab.get_content()
        except Exception as e:
            return f"ERROR: could not read page: {e}"

        text = self.extract_text_with_links(page_source, url)
        self._link_base_url = url
        iframe_text = await self._async_extract_iframe_texts(self._tab, url)
        if iframe_text:
            text = (text or '') + "\n\n" + iframe_text
        frame_target_text = await self._async_extract_frame_target_texts()
        if frame_target_text:
            text = (text or '') + "\n\n" + frame_target_text
        info(f"browse_extract: {len(text)} chars")
        if not text:
            return "ERROR: page has no extractable text."
        return text

    def browse_extract(self, page=None) -> str:
        """The readable text of the current page, paged like fetch_webpage."""
        page = self._page_number(page)
        kept = self._kept_page_text("browse_extract") if page > 1 else None
        if kept is not None:
            info(f"browse_extract: page {page} served from the kept text")
            return self._paged("browse_extract", kept, page)
        text = self._run_browse_op("browse_extract", self._async_browse_extract(),
                                   "ERROR: extraction failed.")
        return self._paged("browse_extract", text, page)

    _BROWSE_EVAL_MAX_CHARS = 8000

    async def _async_browse_eval(self, js: str) -> str:
        if not self._browse_session_url:
            return "ERROR: no browse session. Call browse_open(url) first."
        if err := self._pdf_session_error("no JavaScript context"):
            return err

        body = (js or '').strip()
        if not body:
            return "ERROR: browse_eval needs a JavaScript expression."

        if gate := await self._async_browse_ready_midsession():
            return gate

        # Wrap so the LLM can pass either a bare expression (e.g. window.location.href)
        # or a statement block with explicit `return`. We pick the form at *construction*
        # time: try to build the expression-form AsyncFunction; if that throws a
        # SyntaxError, the body is a statement block, so build the block form instead.
        # The chosen function is then awaited exactly once — distinguishing
        # construction-time SyntaxError from runtime errors is what keeps a
        # non-idempotent body (e.g. a POST) from executing twice. A runtime throw is
        # returned as a tagged value instead of breaking the call.
        body_json = json.dumps(body)
        wrapped = (
            "(async function(){"
            " const AsyncFn = Object.getPrototypeOf(async function(){}).constructor;"
            f" const __body = {body_json};"
            " let __fn;"
            " try { __fn = new AsyncFn('return (' + __body + ');'); }"
            " catch (e) {"
            "   try { __fn = new AsyncFn(__body); }"
            "   catch (e2) {"
            "     const err = e2 && e2.stack || e2;"
            "     return {__floxy_eval_error: String(err)};"
            "   }"
            " }"
            " try { return await __fn(); }"
            " catch (e) {"
            "   const err = e && e.stack || e;"
            "   return {__floxy_eval_error: String(err)};"
            " }"
            "})()"
        )

        try:
            result = await self._evaluate(self._tab, wrapped)
        except Exception as e:
            trace(f"browse_eval evaluate failed: {self._describe_error(e)}")
            return f"ERROR: eval failed: {self._describe_error(e)}"

        dialog_notes = self._take_dialog_notes()
        if isinstance(result, dict) and '__floxy_eval_error' in result:
            err = result['__floxy_eval_error']
            note(f"browse_eval threw in page: {err}")
            return f"{dialog_notes}ERROR: page JS threw: {err}"

        if result is None:
            return f"{dialog_notes}null"
        if isinstance(result, str):
            out = result
        else:
            try:
                out = json.dumps(result, default=str, ensure_ascii=False)
            except Exception as e:
                out = f"<unserializable result: {e}>"

        truncated = False
        if len(out) > self._BROWSE_EVAL_MAX_CHARS:
            out = out[:self._BROWSE_EVAL_MAX_CHARS] + " … (truncated)"
            truncated = True
        info(f"browse_eval: {len(out)} chars{' (truncated)' if truncated else ''}")
        return dialog_notes + out

    def browse_eval(self, js: str) -> str:
        """Run JavaScript in the current browse session and return the result as JSON."""
        info(f"browse_eval: {len((js or '').strip())} char expression")
        trace(f"browse_eval js: {(js or '')[:200]}")
        return self._run_browse_op("browse_eval", self._async_browse_eval(js),
                                   "ERROR: eval failed.")

    @staticmethod
    def _screenshot_dir() -> str:
        return str(LOG_DIR / "screenshots")

    @staticmethod
    def _screenshot_filename(url: str) -> str:
        slug = re.sub(r'[^a-zA-Z0-9._-]+', '_', urlparse(url).netloc + urlparse(url).path)[:60].strip('_') or "page"
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slug}.png"

    async def _async_browse_screenshot(self):
        if not self._browse_session_url:
            return "ERROR: no browse session. Call browse_open(url) first."
        if err := self._pdf_session_error("cannot screenshot"):
            return err

        if gate := await self._async_browse_ready_midsession():
            return gate

        url = self._browse_session_url
        try:
            url = self._tab.target.url or url
        except Exception:
            pass

        try:
            b64 = await self._cdp_wait_for(self._tab.send(cdp.page.capture_screenshot(
                format_="png", capture_beyond_viewport=True
            )), self._EVAL_TIMEOUT)
        except Exception as e:
            trace(f"browse_screenshot: capture_screenshot failed: {self._describe_error(e)}")
            return f"ERROR: screenshot capture failed: {self._describe_error(e)}"

        png_bytes = base64.b64decode(b64)
        geometry = await self._async_page_geometry()

        out_dir = self._screenshot_dir()
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, self._screenshot_filename(url))
        with open(path, "wb") as f:
            f.write(png_bytes)

        self._last_screenshot_bytes = png_bytes
        self._last_screenshot_path = path

        size_kb = max(1, len(png_bytes) // 1024)
        info(f"browse_screenshot: saved {len(png_bytes)} bytes ({size_kb} KB) of {url} to {path}")

        from .ImageAnalyzer import resize_to_jpeg
        llm_bytes = resize_to_jpeg(png_bytes)
        if llm_bytes is None:
            note("browse_screenshot: resize failed, sending original PNG to LLM")
            llm_bytes, media_type = png_bytes, "image/png"
        else:
            media_type = "image/jpeg"
        llm_b64 = base64.b64encode(llm_bytes).decode("ascii")
        info(f"browse_screenshot: sending {len(llm_bytes)} bytes ({media_type}) to LLM")

        return [
            {"type": "text", "text": f"Saved screenshot of {url} to {path} ({size_kb} KB)."
                                     + self._screenshot_scale_note(png_bytes, llm_bytes, geometry)},
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": llm_b64}},
        ]

    async def _async_page_geometry(self) -> Optional[dict]:
        """The viewport, scroll position and device pixel ratio, for saying what a screenshot shows."""
        try:
            geometry = await self._evaluate(
                self._tab,
                "({vw: window.innerWidth, vh: window.innerHeight, dpr: window.devicePixelRatio, "
                "sx: Math.round(window.scrollX), sy: Math.round(window.scrollY)})")
        except Exception as e:
            trace(f"browse_screenshot: page geometry lookup failed: {self._describe_error(e)}")
            return None
        return geometry if isinstance(geometry, dict) else None

    @staticmethod
    def _image_size(data: bytes) -> Optional[tuple]:
        try:
            from PIL import Image
            with Image.open(io.BytesIO(data)) as img:
                return img.size
        except Exception:
            return None

    @classmethod
    def _screenshot_scale_note(cls, png_bytes: bytes, image_bytes: bytes, geometry: Optional[dict]) -> str:
        """What the model needs to click where it sees something: the image is the whole page
        downscaled, so one image pixel is some CSS pixels, and browse_click_at wants viewport
        coordinates, which start at the scroll position. The page's CSS size is the capture's
        pixel size over the device pixel ratio — what was captured, whatever the document
        measures after the capture — and each axis gets its own factor, since the downscale
        rounds the two dimensions separately. Empty when a size is unknown: a missing note
        costs a mis-click, a wrong one costs several."""
        png = cls._image_size(png_bytes)
        size = cls._image_size(image_bytes)
        dpr = (geometry or {}).get('dpr') or 0
        if not png or not size or dpr <= 0:
            return ''
        pw, ph = round(png[0] / dpr), round(png[1] / dpr)
        w, h = size
        fx, fy = pw / w, ph / h
        factors = (f"multiply image coordinates by {fx:.2f}" if f"{fx:.2f}" == f"{fy:.2f}"
                   else f"multiply image x by {fx:.2f} and image y by {fy:.2f}")
        info(f"browse_screenshot: image {w}x{h} px for a {pw}x{ph} CSS px page — "
             f"{fx:.2f} x {fy:.2f} CSS px per image px")
        return (f" The image is {w}x{h} px and shows the whole page, {pw}x{ph} CSS px: {factors} "
                f"to get CSS pixels. browse_click_at takes viewport coordinates (viewport "
                f"{geometry.get('vw')}x{geometry.get('vh')}, scrolled to ({geometry.get('sx')}, "
                f"{geometry.get('sy')})): subtract the scroll offset, and scroll a point outside the "
                "viewport into view first.")

    def browse_screenshot(self, _params=None):
        """Capture a full-page PNG of the current browse session, save it under log/screenshots/, and return the
        image alongside the path so the LLM can both reference and see the page."""
        return self._run_browse_op("browse_screenshot", self._async_browse_screenshot(),
                                   "ERROR: screenshot failed.")

    def get_snapshot_label(self, idx: int) -> Optional[str]:
        """Return the cached label for a snapshot index, or None if not available."""
        if idx < 0 or idx >= len(self._browse_snapshot_labels):
            return None
        return self._browse_snapshot_labels[idx] or None

    def get_last_screenshot(self) -> tuple[Optional[bytes], Optional[str]]:
        """Return (png_bytes, path) for the most recent browse_screenshot, or (None, None) if none cached."""
        return self._last_screenshot_bytes, self._last_screenshot_path

    async def _async_browse_close(self) -> str:
        # Popup targets die with Chrome, so no _async_close_popup_targets pass is
        # needed — reset the Python-side bookkeeping and pull the plug.
        self._reset_browse_session_state()
        await self._async_quit_driver()
        self._finish_browse_close()
        info("browse_close: Chrome fully shut down — a fresh browser starts on the next browse command")
        return "Browse session closed."

    def _finish_browse_close(self):
        """Clear session state and the circuit breaker after the browser is gone."""
        self._browse_session_url = None
        self._browse_pdf_text = None
        self._clear_network_log()
        self._browse_fail_streak = 0
        self._browser_hard_down = False
        self._browser_closed_by_user = True
        # An explicit close is the documented recovery move — the next
        # browse_open must get a real init attempt, not the failure backoff.
        self._last_init_failure_ts = 0.0

    def browse_close(self, _params=None) -> str:
        """Fully shut down the browser (Chrome and all helper processes) and clear session state.

        Reinit is lazy: the next browse command launches a fresh Chrome via the
        health gate. This trades the warm-browser reuse for LLM-predictable
        semantics — close = everything gone, open = fresh browser."""
        info("browse_close: shutting down the browser")
        try:
            return self._loop_thread.submit(self._with_browse_lock(self._async_browse_close()),
                                            timeout=30)
        except Exception as e:
            note(f"browse_close: loop-thread shutdown failed ({type(e).__name__}: {e}) — "
                 "killing tracked Chrome PIDs directly")
            self._reset_browse_session_state()
            self._browser = None
            self._tab = None
            self._kill_tracked_chrome_pids()
            self._finish_browse_close()
            return "Browse session closed."

    # ------------------------------------------------------------------
    # Public sync entry points
    # ------------------------------------------------------------------

    # open_url and browse_extract hand a long page back in pages of this many chars, each
    # headed with its number and the count (a page that fits comes back whole, no header).
    # The text behind a split is kept so a later page is a slice of the same read, not a
    # second fetch — the last few only, keyed by URL (browse_extract by the session).
    PAGE_CHARS = 15000
    _PAGE_TEXTS_KEPT = 8

    @classmethod
    def split_pages(cls, text: str) -> list[str]:
        """The text cut into pages of at most PAGE_CHARS, on a line break where one falls in
        the second half of the page."""
        pages = []
        while len(text) > cls.PAGE_CHARS:
            cut = text.rfind('\n', cls.PAGE_CHARS // 2, cls.PAGE_CHARS)
            if cut < 0:
                cut = cls.PAGE_CHARS
            pages.append(text[:cut])
            text = text[cut:].lstrip('\n')
        pages.append(text)
        return pages

    @staticmethod
    def _page_number(value) -> int:
        """The page a command asked for — 1 unless the parameter is a whole number."""
        try:
            return max(1, int(str(value or '').strip()))
        except ValueError:
            return 1

    def _kept_page_text(self, key: str) -> Optional[str]:
        return getattr(self, '_page_texts', {}).get(key)

    def _paged(self, key: str, text: str, page: int) -> str:
        """`text` as the page asked for. A text that fits in one page is returned as it is;
        a longer one is kept under `key` and the page comes with a header."""
        pages = self.split_pages(text)
        if len(pages) == 1:
            return text
        if not hasattr(self, '_page_texts'):
            self._page_texts = {}
        self._page_texts.pop(key, None)
        self._page_texts[key] = text
        while len(self._page_texts) > self._PAGE_TEXTS_KEPT:
            del self._page_texts[next(iter(self._page_texts))]
        if page > len(pages):
            info(f"Page {page} asked of {key}, which has {len(pages)} pages")
            return f"ERROR: the page has {len(pages)} pages, there is no page {page}."
        info(f"Page {page} of {len(pages)} ({len(pages[page - 1])} of {len(text)} chars) for {key}")
        return f"[Page {page} of {len(pages)} — {len(text)} chars in all]\n{pages[page - 1]}"

    def _resolve_page_link(self, url: str) -> str:
        """A root-relative path, as the extractor renders a same-site link, made absolute
        against the page it was read from; any other URL as it is."""
        url = (url or '').strip()
        base = getattr(self, '_link_base_url', None)
        if url.startswith('/') and not url.startswith('//') and base:
            resolved = urljoin(base, url)
            info(f"Link {url} resolved against {base} to {resolved}")
            return resolved
        return url

    def fetch_webpage(self, url: str, page=1) -> str:
        page = self._page_number(page)
        key = self._resolve_page_link(url)
        url = key
        kept = self._kept_page_text(key) if page > 1 else None
        if kept is not None:
            info(f"Page {page} of {key} served from the kept text")
            return self._paged(key, kept, page)
        return self._paged(key, self._fetch_webpage_text(url), page)

    def _fetch_webpage_text(self, url: str) -> str:
        try:
            url = self._normalize_url(url)
            self._link_base_url = url

            info(f"Download content from {url}...")

            if self._is_pdf_url(url):
                return self._fetch_pdf_content(url)

            # Tier 1: nodriver (best for JS-heavy sites)
            page_source = self._fetch_via_chrome(url)
            if page_source and not self._is_blocked_response(page_source):
                full_text = self.extract_text_with_links(page_source, url)
                if full_text:
                    labels = []
                    try:
                        labels = self._loop_thread.submit(self._async_tab_labels(self._tab), timeout=10) or []
                    except Exception as e:
                        trace(f"tab-labels submit failed: {e}")
                    tab_note = self.tab_note(labels)
                    if tab_note:
                        info(f"Page has a tab strip of {len(labels)} — panels not read, tabs named: {labels}")
                        full_text = full_text + "\n\n" + tab_note
                    else:
                        trace("No tab strip on the page")

                    iframe_text = ''
                    try:
                        iframe_text = self._loop_thread.submit(
                            self._async_extract_iframe_texts(self._tab, url),
                            timeout=30,
                        ) or ''
                    except Exception as e:
                        trace(f"iframe-text submit failed: {e}")
                    if iframe_text:
                        full_text = full_text + "\n\n" + iframe_text
                        info(f"Appended {len(iframe_text)} chars from iframes")
                    info(f"Downloaded {len(full_text)} chars via nodriver.")
                    return full_text

            # Tier 2: cloudscraper (handles Cloudflare challenges)
            note(f"nodriver result looks blocked for {url}, trying cloudscraper...")
            cs_html = self._fetch_via_cloudscraper(url)
            if cs_html and not self._is_blocked_response(cs_html):
                full_text = self.extract_text_with_links(cs_html, url)
                if full_text:
                    info(f"Downloaded {len(full_text)} chars via cloudscraper.")
                    return full_text

            # Tier 3: curl_cffi (handles DataDome/TLS fingerprint blocks)
            note(f"cloudscraper also blocked for {url}, trying curl_cffi...")
            cffi_html = self._fetch_via_curl_cffi(url)
            if cffi_html and not self._is_blocked_response(cffi_html):
                full_text = self.extract_text_with_links(cffi_html, url)
                if full_text:
                    info(f"Downloaded {len(full_text)} chars via curl_cffi.")
                    return full_text

            # If nodriver returned something (possibly partial), use it as last resort
            if page_source:
                full_text = self.extract_text_with_links(page_source, url)
                if full_text:
                    info(f"Using possibly partial nodriver result ({len(full_text)} chars).")
                    return full_text

            info("Could not download any content from site!")
            return "ERROR: Could not fetch this URL. Please try another one."
        except Exception as e:
            e_ = "Error when fetching webpage: " + str(e)
            note(e_)
            return e_

    def search(self, query: str) -> str:
        """Google via the Serper API — the only search backend, so a missing key is reported
        to the LLM as such (it can ask for the key through config_request)."""
        query = (query or "").strip()
        if not query:
            note("Empty search query received.")
            return "No search query provided."

        config = ConfigManager()
        try:
            api_key = config.get_api_key('serper')
        except KeyError:
            api_key = None

        if not api_key or 'YOUR_' in api_key:
            note("Serper API key not configured, web search unavailable.")
            return ("ERROR: Web search needs api_keys.serper in config.yaml "
                    "(a free key from https://serper.dev).")

        country_hint = config.get('search.country_hint', 'se')
        interface_language = config.get('search.interface_language', 'en')
        raw_limit = config.get('search.results_limit', 6)
        try:
            result_limit = max(1, min(int(raw_limit), 10))
        except (TypeError, ValueError):
            result_limit = 6

        payload = {'q': query, 'num': result_limit}
        if country_hint:
            payload['gl'] = country_hint
        if interface_language:
            payload['hl'] = interface_language

        info(f"Searching via Serper API for '{query}'")

        try:
            response = requests.post(
                'https://google.serper.dev/search',
                headers={'X-API-KEY': api_key, 'Content-Type': 'application/json'},
                json=payload,
                timeout=10
            )
            response.raise_for_status()

            organic = response.json().get('organic', [])
            if not organic:
                info("No organic results from Serper API.")
                return f"No search results found for: {query}"

            formatted = "Google search results:\n\n"
            for item in organic[:result_limit]:
                title = item.get('title', 'No title')
                link = item.get('link', '')
                snippet = item.get('snippet', '')
                formatted += f"{title}\n{link}\n{snippet}\n\n"

            info(f"Found {min(len(organic), result_limit)} results via Serper API.")
            trace(formatted)
            return formatted.strip()

        except requests.exceptions.RequestException as e:
            error_msg = f"Error calling Serper API: {e}"
            note(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error during search: {e}"
            note(error_msg)
            return error_msg
