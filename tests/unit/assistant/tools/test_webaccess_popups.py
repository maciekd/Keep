"""Popup/tab target bookkeeping (plan 02): gated adoption, pending blank targets,
verified restores, and transport-failure no-mutation semantics."""
import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from assistant.tools.WebAccess import WebAccess
from nodriver import cdp
from tests.unit.assistant.tools.virtual_time import virtual_time


class FakeTargetInfo:
    def __init__(self, target_id, type_='page', url=''):
        self.target_id = cdp.target.TargetID(target_id)
        self.type_ = type_
        self.url = url


class FakeTab:
    def __init__(self, target_id, url='', type_='page'):
        self.target = FakeTargetInfo(target_id, type_, url)

    async def evaluate(self, *args, **kwargs):  # satisfies _as_tab's Tab check
        return None


class FakeConnection:
    """Answers Target.getTargets from the browser's live-target list and records
    Target.closeTarget calls (removing the closed target from the live list)."""

    def __init__(self, browser):
        self._browser = browser
        self.closed = []

    async def send(self, cdp_obj, **kwargs):
        name = getattr(getattr(cdp_obj, 'gi_code', None), 'co_name', '')
        if name == 'get_targets':
            if self._browser.get_targets_error is not None:
                raise self._browser.get_targets_error
            return [t.target for t in self._browser.live_targets]
        if name == 'close_target':
            tid = str(cdp_obj.send(None)['params']['targetId'])
            self.closed.append(tid)
            self._browser.live_targets = [
                t for t in self._browser.live_targets if str(t.target.target_id) != tid
            ]
            return None
        return None


class FakeBrowser:
    def __init__(self, targets, live=None):
        self.targets = targets  # nodriver registry view (used by _page_targets/_as_tab)
        self.live_targets = list(targets) if live is None else list(live)  # getTargets view
        self.get_targets_error = None
        self.connection = FakeConnection(self)

    async def update_targets(self):
        pass


def make_wa(base_tab, browser, grace=0.0):
    wa = WebAccess.__new__(WebAccess)
    wa._browser = browser
    wa._tab = base_tab
    wa._popup_parents = []
    wa._known_page_targets = {str(base_tab.target.target_id)} if base_tab else set()
    wa._pending_target_notes = []
    wa._dialog_notes = []
    wa._pending_popup_targets = {}
    wa._browse_frame_sessions = {}
    wa._browse_snapshot_frame_ids = []
    wa._browse_snapshot_labels = []
    wa._last_screenshot_bytes = None
    wa._last_screenshot_path = None
    wa._POPUP_BLANK_GRACE = grace
    wa._POPUP_BLANK_POLL = 0.02
    return wa


def run(coro):
    return asyncio.run(coro)


def idle_patch():
    return patch.object(WebAccess, '_async_wait_for_page_ready', new=AsyncMock())


@pytest.mark.unit
class TestLivePageTargets:

    def test_filters_page_targets(self):
        base = FakeTab('base', 'https://shop.example')
        iframe = FakeTab('if1', 'https://pay.klarna.com/frame', type_='iframe')
        wa = make_wa(base, FakeBrowser([base, iframe]))
        live = run(wa._async_live_page_targets())
        assert set(live) == {'base'}

    def test_returns_none_on_transport_failure(self):
        base = FakeTab('base', 'https://shop.example')
        browser = FakeBrowser([base])
        browser.get_targets_error = ConnectionError('Errno 61')
        wa = make_wa(base, browser)
        assert run(wa._async_live_page_targets()) is None

@pytest.mark.unit
class TestGatedAdoption:

    def test_real_url_popup_adopted(self):
        # Case 1: regression baseline for Klarna-style popups.
        base = FakeTab('base', 'https://shop.example/kassa')
        popup = FakeTab('pop1', 'https://bank.example/pay')
        wa = make_wa(base, FakeBrowser([base, popup]))
        with idle_patch():
            run(wa._async_sync_popup_targets())
        assert wa._tab is popup
        assert wa._popup_parents == [base]
        assert wa._pending_popup_targets == {}
        assert len(wa._pending_target_notes) == 1
        assert 'https://bank.example/pay' in wa._pending_target_notes[0]

    def test_blank_target_goes_pending_then_adopted_after_navigation(self):
        # Case 2a: fresh about:blank target is not adopted; it is adopted on a
        # later sync once it has navigated to a real URL.
        base = FakeTab('base', 'https://shop.example')
        popup = FakeTab('pop1', 'about:blank')
        wa = make_wa(base, FakeBrowser([base, popup]))
        run(wa._async_sync_popup_targets())
        assert wa._tab is base
        assert wa._popup_parents == []
        assert wa._pending_target_notes == []
        assert 'pop1' in wa._pending_popup_targets
        assert 'pop1' in wa._known_page_targets

        popup.target.url = 'https://bank.example/pay'
        with idle_patch():
            run(wa._async_sync_popup_targets())
        assert wa._tab is popup
        assert wa._popup_parents == [base]
        assert wa._pending_popup_targets == {}
        assert len(wa._pending_target_notes) == 1

    def test_blank_target_that_disappears_is_dropped_silently(self):
        # Case 2b.
        base = FakeTab('base', 'https://shop.example')
        popup = FakeTab('pop1', 'about:blank')
        browser = FakeBrowser([base, popup])
        wa = make_wa(base, browser)
        run(wa._async_sync_popup_targets())
        assert 'pop1' in wa._pending_popup_targets

        browser.targets = [base]
        browser.live_targets = [base]
        run(wa._async_sync_popup_targets())
        assert wa._tab is base
        assert wa._pending_popup_targets == {}
        assert wa._pending_target_notes == []

    def test_blank_target_expires_from_pending(self):
        base = FakeTab('base', 'https://shop.example')
        popup = FakeTab('pop1', 'about:blank')
        wa = make_wa(base, FakeBrowser([base, popup]))
        wa._known_page_targets = {'base', 'pop1'}
        wa._pending_popup_targets = {'pop1': 0.0}  # far past the 10s expiry
        run(wa._async_sync_popup_targets())
        assert wa._tab is base
        assert wa._pending_popup_targets == {}
        assert wa._pending_target_notes == []

    def test_inline_grace_adopts_popup_that_navigates_quickly(self):
        # The browse_act -> window.open case: blank at first sight, navigates
        # within the inline grace window, adopted in the same sync call.
        base = FakeTab('base', 'https://shop.example')
        popup = FakeTab('pop1', 'about:blank')
        wa = make_wa(base, FakeBrowser([base, popup]), grace=1.0)

        async def scenario():
            async def navigate():
                await asyncio.sleep(0.03)
                popup.target.url = 'https://bank.example/pay'
            task = asyncio.create_task(navigate())
            with idle_patch():
                await wa._async_sync_popup_targets()
            await task

        with virtual_time():
            run(scenario())
        assert wa._tab is popup
        assert wa._popup_parents == [base]
        assert wa._pending_popup_targets == {}

    def test_grace_poll_target_vanishing_is_not_parked_as_pending(self):
        # The None return of _async_wait_for_fresh_target_url: a blank target
        # that CLOSES during the grace poll must be forgotten, not parked in the
        # pending set where it would sit until the 10s expiry.
        base = FakeTab('base', 'https://shop.example')
        popup = FakeTab('pop1', 'about:blank')
        browser = FakeBrowser([base, popup])
        wa = make_wa(base, browser, grace=1.0)

        async def scenario():
            async def vanish():
                await asyncio.sleep(0.03)
                browser.targets = [base]
                browser.live_targets = [base]
            task = asyncio.create_task(vanish())
            await wa._async_sync_popup_targets()
            await task

        with virtual_time():
            run(scenario())
        assert wa._tab is base
        assert wa._pending_popup_targets == {}
        assert wa._pending_target_notes == []

    def test_grace_poll_transport_death_defers_to_reinit_not_pending(self):
        # Transport dying mid-grace also yields None: parking the target (or
        # keeping the poll spinning) would fight the dead-browser reinit path
        # that owns recovery.
        base = FakeTab('base', 'https://shop.example')
        popup = FakeTab('pop1', 'about:blank')
        browser = FakeBrowser([base, popup])
        wa = make_wa(base, browser, grace=1.0)

        async def scenario():
            async def kill_transport():
                await asyncio.sleep(0.03)
                browser.get_targets_error = ConnectionError('Errno 61')
            task = asyncio.create_task(kill_transport())
            await wa._async_sync_popup_targets()
            await task

        with virtual_time():
            run(scenario())
        assert wa._tab is base
        assert wa._pending_popup_targets == {}
        assert wa._pending_target_notes == []

    def test_reinit_seeded_state_adopts_nothing(self):
        # Case 3: a production bug — after reinit the base tab and a stray blank
        # target must not be adopted as popups, even with an unseeded known set.
        base = FakeTab('base', 'https://shop.example/search')
        stray = FakeTab('stray', 'about:blank')
        wa = make_wa(base, FakeBrowser([base, stray]))
        wa._known_page_targets = set()  # worst case: seeding was lost
        run(wa._async_sync_popup_targets())
        assert wa._tab is base
        assert wa._popup_parents == []
        assert wa._pending_target_notes == []
        assert 'stray' in wa._pending_popup_targets
        assert wa._known_page_targets == {'base', 'stray'}

        # And with the post-init seeding in place nothing is even pending.
        wa2 = make_wa(base, FakeBrowser([base, stray]))
        wa2._known_page_targets = {'base', 'stray'}
        run(wa2._async_sync_popup_targets())
        assert wa2._tab is base
        assert wa2._popup_parents == []
        assert wa2._pending_popup_targets == {}
        assert wa2._pending_target_notes == []


@pytest.mark.unit
class TestVerifiedRestore:

    def test_close_with_dead_base_adopts_newest_live_target(self):
        # Case 4.
        base = FakeTab('base', 'https://shop.example')
        popup = FakeTab('pop1', 'https://bank.example/pay')
        newtab = FakeTab('new1', 'https://shop.example/confirmation')
        browser = FakeBrowser([base, popup, newtab], live=[popup, newtab])
        wa = make_wa(base, browser)
        wa._tab = popup
        wa._popup_parents = [base]
        run(wa._async_close_popup_targets())
        assert browser.connection.closed == ['pop1']  # nothing sent for the dead base
        assert wa._tab is newtab
        assert wa._popup_parents == []
        assert len(wa._pending_target_notes) == 1
        assert 'https://shop.example/confirmation' in wa._pending_target_notes[0]
        assert wa._known_page_targets == {'new1'}

    def test_close_with_everything_dead_defers_to_reinit(self):
        # Approved decision: no fresh about:blank tab — clear the tab and let
        # the package 1 reinit path rebuild the browser.
        base = FakeTab('base', 'https://shop.example')
        popup = FakeTab('pop1', 'https://bank.example/pay')
        browser = FakeBrowser([base, popup], live=[])
        wa = make_wa(base, browser)
        wa._tab = popup
        wa._popup_parents = [base]
        run(wa._async_close_popup_targets())
        assert wa._tab is None
        assert wa._popup_parents == []
        assert browser.connection.closed == []

    def test_close_transport_failure_mutates_nothing(self):
        # Case 5a.
        base = FakeTab('base', 'https://shop.example')
        popup = FakeTab('pop1', 'https://bank.example/pay')
        browser = FakeBrowser([base, popup])
        browser.get_targets_error = ConnectionError('Errno 61')
        wa = make_wa(base, browser)
        wa._tab = popup
        wa._popup_parents = [base]
        wa._known_page_targets = {'base', 'pop1'}
        run(wa._async_close_popup_targets())
        assert wa._tab is popup
        assert wa._popup_parents == [base]
        assert wa._known_page_targets == {'base', 'pop1'}
        assert browser.connection.closed == []

    def test_sync_transport_failure_mutates_nothing(self):
        # Case 5b.
        base = FakeTab('base', 'https://shop.example')
        popup = FakeTab('pop1', 'https://bank.example/pay')
        browser = FakeBrowser([base, popup])
        browser.get_targets_error = ConnectionError('Errno 61')
        wa = make_wa(base, browser)
        wa._tab = popup
        wa._popup_parents = [base]
        wa._known_page_targets = {'base', 'pop1'}
        wa._pending_popup_targets = {'p': 1.0}
        run(wa._async_sync_popup_targets())
        assert wa._tab is popup
        assert wa._popup_parents == [base]
        assert wa._known_page_targets == {'base', 'pop1'}
        assert wa._pending_popup_targets == {'p': 1.0}
        assert wa._pending_target_notes == []


@pytest.mark.unit
class TestSyncFallback:

    def test_all_dead_adopts_newest_live_target(self):
        # Case 6.
        base = FakeTab('base', 'https://shop.example')
        p1 = FakeTab('pop1', 'https://bank.example/step1')
        p2 = FakeTab('pop2', 'https://bankid.example/auth')
        newtab = FakeTab('new1', 'https://shop.example/confirmation')
        browser = FakeBrowser([base, p1, p2, newtab], live=[newtab])
        wa = make_wa(base, browser)
        wa._tab = p2
        wa._popup_parents = [base, p1]
        wa._known_page_targets = {'base', 'pop1', 'pop2', 'new1'}
        run(wa._async_sync_popup_targets())
        assert wa._tab is newtab
        assert wa._popup_parents == []
        assert any('https://shop.example/confirmation' in n for n in wa._pending_target_notes)
        assert wa._known_page_targets == {'new1'}

    def test_all_dead_recovery_skips_browser_chrome_targets(self):
        # The newest live target is browser chrome — adopting devtools as the
        # new base tab would silently point every later snapshot/act at the
        # inspector instead of a page.
        base = FakeTab('base', 'https://shop.example')
        newtab = FakeTab('new1', 'https://shop.example/confirmation')
        devtools = FakeTab('dt1', 'devtools://devtools/inspector.html')
        browser = FakeBrowser([base, newtab, devtools], live=[newtab, devtools])
        wa = make_wa(base, browser)
        wa._known_page_targets = {'base', 'new1', 'dt1'}
        run(wa._async_sync_popup_targets())
        assert wa._tab is newtab
        assert wa._popup_parents == []
        assert any('https://shop.example/confirmation' in n for n in wa._pending_target_notes)

    def test_all_dead_with_only_chrome_targets_defers_to_reinit(self):
        # Only browser-chrome targets left alive: same as nothing left — clear
        # the tab so the reinit path rebuilds the browser instead of adopting
        # devtools as a ghost base tab.
        base = FakeTab('base', 'https://shop.example')
        devtools = FakeTab('dt1', 'devtools://devtools/inspector.html')
        browser = FakeBrowser([base, devtools], live=[devtools])
        wa = make_wa(base, browser)
        wa._known_page_targets = {'base', 'dt1'}
        run(wa._async_sync_popup_targets())
        assert wa._tab is None
        assert wa._popup_parents == []
        assert wa._pending_popup_targets == {}

    def test_all_dead_with_no_live_target_clears_tab(self):
        base = FakeTab('base', 'https://shop.example')
        p1 = FakeTab('pop1', 'https://bank.example/step1')
        browser = FakeBrowser([base, p1], live=[])
        wa = make_wa(base, browser)
        wa._tab = p1
        wa._popup_parents = [base]
        wa._known_page_targets = {'base', 'pop1'}
        run(wa._async_sync_popup_targets())
        assert wa._tab is None
        assert wa._popup_parents == []
        assert wa._known_page_targets == set()
        assert wa._pending_popup_targets == {}


@pytest.mark.unit
class TestAdoptionBookkeepingGuards:

    def test_live_but_unregistered_target_defers_adoption(self):
        # Verified live via CDP but not yet in nodriver's registry (popup race):
        # adoption must park the target in the pending set and keep the current
        # tab, not switch to a target _as_tab cannot build yet.
        base = FakeTab('base', 'https://shop.example')
        popup = FakeTab('pop1', 'https://bank.example/pay')
        wa = make_wa(base, FakeBrowser([base], live=[base, popup]))
        run(wa._async_sync_popup_targets())
        assert wa._tab is base
        assert wa._popup_parents == []
        assert 'pop1' in wa._pending_popup_targets
        assert wa._pending_target_notes == []

    def test_active_tabs_appearing_fresh_are_not_readopted(self):
        # After known-target reseeding the active popup and its parent both show
        # up as "fresh" — re-adopting them would push the active tab onto its own
        # parent stack (self-parenting).
        base = FakeTab('base', 'https://shop.example')
        popup = FakeTab('pop1', 'https://bank.example/pay')
        wa = make_wa(popup, FakeBrowser([base, popup]))
        wa._popup_parents = [base]
        wa._known_page_targets = set()
        run(wa._async_sync_popup_targets())
        assert wa._tab is popup
        assert wa._popup_parents == [base]
        assert wa._pending_target_notes == []
        assert wa._pending_popup_targets == {}
