"""Package 3: real keyboard input primitives — key table, CDP dispatch,
CDP-backed fill with legacy fallback, browse_act's \\n marker, browse_key."""
import asyncio
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, patch
from nodriver.core.connection import ProtocolException

from assistant.tools.WebAccess import WebAccess


def make_wa(**overrides):
    wa = WebAccess.__new__(WebAccess)
    wa._browser = None
    wa._tab = SimpleNamespace(send=AsyncMock())
    wa._browse_session_url = 'https://shop.example'
    wa._browse_pdf_text = None
    wa._browse_frame_sessions = {}
    wa._browse_snapshot_frame_ids = [None]
    wa._browse_snapshot_labels = []
    wa._pending_target_notes = []
    wa._dialog_notes = []
    for k, v in overrides.items():
        setattr(wa, k, v)
    return wa


def run(coro):
    return asyncio.run(coro)


def sent_payload(call):
    """Unwrap the CDP generator handed to a fake send/_send_on_session call."""
    return next(call.args[0])


@pytest.mark.unit
class TestKeyTable:

    def test_alias_resolution(self):
        assert WebAccess._resolve_key('esc') == WebAccess._KEY_DEFS['escape']
        assert WebAccess._resolve_key('return') == WebAccess._KEY_DEFS['enter']
        assert WebAccess._resolve_key('down') == WebAccess._KEY_DEFS['arrowdown']
        assert WebAccess._resolve_key('up') == WebAccess._KEY_DEFS['arrowup']
        assert WebAccess._resolve_key('left') == WebAccess._KEY_DEFS['arrowleft']
        assert WebAccess._resolve_key('right') == WebAccess._KEY_DEFS['arrowright']

    def test_case_insensitive_and_whitespace(self):
        assert WebAccess._resolve_key('ENTER') == WebAccess._KEY_DEFS['enter']
        assert WebAccess._resolve_key(' ArrowDown ') == WebAccess._KEY_DEFS['arrowdown']

    def test_unknown_key_is_none(self):
        assert WebAccess._resolve_key('F5') is None
        assert WebAccess._resolve_key('') is None
        assert WebAccess._resolve_key(None) is None


@pytest.mark.unit
class TestDispatchKey:

    def test_enter_sends_keydown_with_text_then_keyup_without(self):
        wa = make_wa()
        run(wa._async_dispatch_key('enter'))
        assert wa._tab.send.await_count == 2
        down = sent_payload(wa._tab.send.await_args_list[0])
        up = sent_payload(wa._tab.send.await_args_list[1])
        assert down['method'] == 'Input.dispatchKeyEvent'
        assert down['params']['type'] == 'keyDown'
        assert down['params']['text'] == '\r'
        assert down['params']['unmodifiedText'] == '\r'
        assert down['params']['windowsVirtualKeyCode'] == 13
        assert up['params']['type'] == 'keyUp'
        assert 'text' not in up['params']

    def test_escape_carries_no_text_on_keydown(self):
        wa = make_wa()
        run(wa._async_dispatch_key('Escape'))
        down = sent_payload(wa._tab.send.await_args_list[0])
        assert down['params']['type'] == 'keyDown'
        assert 'text' not in down['params']
        assert down['params']['windowsVirtualKeyCode'] == 27

    def test_unknown_key_raises(self):
        wa = make_wa()
        with pytest.raises(ValueError, match='unsupported key'):
            run(wa._async_dispatch_key('F5'))
        wa._tab.send.assert_not_awaited()

    def test_frame_id_routes_through_frame_session(self):
        wa = make_wa()
        send_on_session = AsyncMock()
        with patch.object(WebAccess, '_async_attach_frame_session',
                          new=AsyncMock(return_value='sess-1')), \
                patch.object(WebAccess, '_send_on_session', new=send_on_session):
            run(wa._async_dispatch_key('enter', frame_id='ft1'))
        assert send_on_session.await_count == 2
        assert all(call.args[1] == 'sess-1' for call in send_on_session.await_args_list)
        wa._tab.send.assert_not_awaited()

    def test_frame_session_failure_falls_back_to_main_tab(self):
        wa = make_wa()
        with patch.object(WebAccess, '_async_attach_frame_session',
                          new=AsyncMock(return_value='sess-1')), \
                patch.object(WebAccess, '_send_on_session',
                             new=AsyncMock(side_effect=ProtocolException('no session'))):
            run(wa._async_dispatch_key('enter', frame_id='ft1'))
        assert wa._tab.send.await_count == 2
        down = sent_payload(wa._tab.send.await_args_list[0])
        assert down['params']['type'] == 'keyDown'


@pytest.fixture(autouse=True)
def no_popup_settle(monkeypatch):
    """The post-action settle is real time Chrome needs, not behaviour under test: 0.3 s × 13
    tests was the slowest file in the suite."""
    monkeypatch.setattr(WebAccess, "_POPUP_SETTLE_S", 0)
    monkeypatch.setattr(WebAccess, "_async_action_mark", AsyncMock(return_value=None))
    monkeypatch.setattr(WebAccess, "_async_wait_for_page_change", AsyncMock(return_value=('changed', None)))


def act_patches(execute=None):
    return (
        patch.object(WebAccess, '_async_browse_ready_midsession',
                     new=AsyncMock(return_value=None)),
        patch.object(WebAccess, '_async_execute_action',
                     new=execute or AsyncMock(return_value=None)),
        patch.object(WebAccess, '_async_sync_popup_targets', new=AsyncMock()),
        patch.object(WebAccess, '_async_wait_for_page_ready', new=AsyncMock()),
        patch.object(WebAccess, '_async_take_snapshot',
                     new=AsyncMock(return_value='SNAP')),
    )


@pytest.mark.unit
class TestBrowseActEnterMarker:
    """The LLM's "...\\n" arrives as LITERAL backslash-n through the command
    parser (no \\n escape resolution) — that is the primary marker; a real
    trailing newline is honored defensively."""

    def _act(self, spec):
        wa = make_wa(_browse_snapshot_frame_ids=[None, None, None])
        execute = AsyncMock(return_value=None)
        p = act_patches(execute)
        with p[0], p[1], p[2], p[3], p[4]:
            result = run(wa._async_browse_act(spec))
        return result, execute

    def test_literal_backslash_n_sets_press_enter_and_strips_marker(self):
        result, execute = self._act('2=foo\\n')
        assert result == 'SNAP'
        execute.assert_awaited_once_with(2, 'foo', True)

    def test_plain_fill_does_not_press_enter(self):
        _, execute = self._act('2=foo')
        execute.assert_awaited_once_with(2, 'foo', False)

    def test_marker_only_fills_empty_and_presses_enter(self):
        _, execute = self._act('2=\\n')
        execute.assert_awaited_once_with(2, '', True)

    def test_real_trailing_newline_also_counts(self):
        _, execute = self._act('2=foo\n')
        execute.assert_awaited_once_with(2, 'foo', True)

    def test_click_spec_never_presses_enter(self):
        _, execute = self._act('2')
        execute.assert_awaited_once_with(2, None, False)


@pytest.mark.unit
class TestSplitActBatch:

    def test_splits_fills_and_trailing_click(self):
        assert WebAccess._split_act_batch('4=2026-05-02; 5=4; 6=19:00; 7') == \
            ['4=2026-05-02', '5=4', '6=19:00', '7']

    def test_semicolon_inside_value_stays_in_value(self):
        assert WebAccess._split_act_batch('3=red; blue; 4=x') == ['3=red; blue', '4=x']

    def test_consecutive_clicks(self):
        assert WebAccess._split_act_batch('5; 7') == ['5', '7']


@pytest.mark.unit
class TestBrowseActBatch:

    def _wa(self, n_elements=8):
        return make_wa(_browse_snapshot_frame_ids=[None] * n_elements)

    def test_batch_executes_in_order_with_one_snapshot(self):
        wa = self._wa()
        execute = AsyncMock(return_value=None)
        p = act_patches(execute)
        with p[0], p[1], p[2], p[3], p[4] as snap:
            result = run(wa._async_browse_act('4=a; 5=b; 6'))
        assert result == 'SNAP'
        assert [c.args for c in execute.await_args_list] == \
            [(4, 'a', False), (5, 'b', False), (6, None, False)]
        snap.assert_awaited_once()

    def test_enter_marker_applies_per_op(self):
        wa = self._wa()
        execute = AsyncMock(return_value=None)
        p = act_patches(execute)
        with p[0], p[1], p[2], p[3], p[4]:
            run(wa._async_browse_act('2=foo\\n; 3=bar'))
        assert [c.args for c in execute.await_args_list] == \
            [(2, 'foo', True), (3, 'bar', False)]

    def test_real_trailing_newline_applies_to_last_op_only(self):
        wa = self._wa()
        execute = AsyncMock(return_value=None)
        p = act_patches(execute)
        with p[0], p[1], p[2], p[3], p[4]:
            run(wa._async_browse_act('2=foo; 3=bar\n'))
        assert [c.args for c in execute.await_args_list] == \
            [(2, 'foo', False), (3, 'bar', True)]

    def test_out_of_range_index_anywhere_aborts_before_executing(self):
        wa = self._wa(n_elements=3)
        execute = AsyncMock(return_value=None)
        p = act_patches(execute)
        with p[0], p[1], p[2], p[3], p[4]:
            result = run(wa._async_browse_act('1=a; 99=b'))
        assert 'index 99 is out of range' in result
        assert 'No batched action was executed' in result
        execute.assert_not_awaited()

    def test_failure_mid_batch_skips_rest_and_reports_progress(self):
        wa = self._wa()
        execute = AsyncMock(side_effect=[None, 'element no longer in DOM', None])
        p = act_patches(execute)
        with p[0], p[1], p[2], p[3], p[4]:
            result = run(wa._async_browse_act('1=a; 2=b; 3=c'))
        assert 'ERROR: action on [2] failed: element no longer in DOM' in result
        executed, skipped = result.split("before the failure")
        assert "'1=" in executed and "'3=" in skipped
        assert "=a" not in result and "=c" not in result, "a typed value is in no note: the log holds the note too"
        assert execute.await_count == 2

    def test_navigation_mid_batch_skips_rest_with_warning(self):
        wa = self._wa()
        execute = AsyncMock(return_value=None)
        urls = iter(['https://shop.example/form',   # start
                     'https://shop.example/form',   # after op 1: unchanged
                     'https://shop.example/done'])  # after op 2: navigated
        p = act_patches(execute)
        with p[0], p[1], p[2], p[3], p[4] as snap, \
                patch.object(WebAccess, '_current_tab_url',
                             side_effect=lambda self: next(urls), autospec=True):
            result = run(wa._async_browse_act('1=a; 2; 3=c'))
        assert "WARNING: action '2' navigated" in result
        executed, skipped = result.split("; skipped ")
        assert "'1=" in executed and "'2'" in executed and "'3=" in skipped
        assert "=a" not in result and "=c" not in result
        assert result.endswith('SNAP')
        assert execute.await_count == 2
        snap.assert_awaited_once()

    def test_query_param_change_does_not_stop_batch(self):
        # Shopify variant selects rewrite ?variant=… — same host+path, not a navigation.
        wa = self._wa()
        execute = AsyncMock(return_value=None)
        urls = iter(['https://shop.example/p', 'https://shop.example/p?variant=2'])
        p = act_patches(execute)
        with p[0], p[1], p[2], p[3], p[4], \
                patch.object(WebAccess, '_current_tab_url',
                             side_effect=lambda self: next(urls), autospec=True):
            result = run(wa._async_browse_act('1=Black; 2'))
        assert result == 'SNAP'
        assert execute.await_count == 2


META_TEXT = {'tag': 'input', 'type': 'text', 'role': '', 'contenteditable': None, 'checked': None}
META_BUTTON = {'tag': 'button', 'type': '', 'role': '', 'contenteditable': None, 'checked': None}


@pytest.mark.unit
class TestCdpFill:

    def test_main_page_fill_uses_insert_text_and_commits(self):
        wa = make_wa()
        # meta, scroll, focus-clear, commit
        main_eval = AsyncMock(side_effect=[META_TEXT, None, True, True])
        with patch.object(WebAccess, '_evaluate', new=main_eval), \
                patch.object(WebAccess, '_async_dispatch_key', new=AsyncMock()) as dispatch:
            assert run(wa._async_execute_action(0, 'foo')) is None
        sends = wa._tab.send.await_args_list
        assert len(sends) == 1
        payload = sent_payload(sends[0])
        assert payload == {'method': 'Input.insertText', 'params': {'text': 'foo'}}
        # no Enter requested: change+blur commit ran, no key dispatched
        assert 'blur()' in main_eval.await_args_list[3].args[1]
        dispatch.assert_not_awaited()

    def test_press_enter_dispatches_before_any_blur(self):
        wa = make_wa()
        main_eval = AsyncMock(side_effect=[META_TEXT, None, True])
        with patch.object(WebAccess, '_evaluate', new=main_eval), \
                patch.object(WebAccess, '_async_dispatch_key', new=AsyncMock()) as dispatch:
            assert run(wa._async_execute_action(0, 'foo', press_enter=True)) is None
        dispatch.assert_awaited_once_with('enter', None)
        # commit (change+blur) must NOT run when Enter is pressed
        assert main_eval.await_count == 3
        assert not any('blur()' in c.args[1] for c in main_eval.await_args_list)

    def test_insert_text_failure_falls_back_to_legacy_js_fill(self):
        wa = make_wa()
        wa._tab.send = AsyncMock(side_effect=RuntimeError('insertText rejected'))
        # meta, scroll, focus-clear, legacy fill
        main_eval = AsyncMock(side_effect=[META_TEXT, None, True, True])
        with patch.object(WebAccess, '_evaluate', new=main_eval):
            assert run(wa._async_execute_action(0, 'foo')) is None
        assert ", 'foo');" in main_eval.await_args_list[3].args[1]

    def test_fallback_still_attempts_enter(self):
        wa = make_wa()
        wa._tab.send = AsyncMock(side_effect=RuntimeError('insertText rejected'))
        main_eval = AsyncMock(side_effect=[META_TEXT, None, True, True])
        with patch.object(WebAccess, '_evaluate', new=main_eval), \
                patch.object(WebAccess, '_async_dispatch_key', new=AsyncMock()) as dispatch:
            assert run(wa._async_execute_action(0, 'foo', press_enter=True)) is None
        dispatch.assert_awaited_once_with('enter', None)

    def test_legacy_fill_failure_reports_error(self):
        wa = make_wa()
        wa._tab.send = AsyncMock(side_effect=RuntimeError('insertText rejected'))
        main_eval = AsyncMock(side_effect=[META_TEXT, None, True, RuntimeError('boom')])
        with patch.object(WebAccess, '_evaluate', new=main_eval):
            err = run(wa._async_execute_action(0, 'foo'))
        assert err == 'fill failed: boom'

    def test_press_enter_on_non_text_element_errors(self):
        wa = make_wa()
        main_eval = AsyncMock(side_effect=[META_BUTTON])
        with patch.object(WebAccess, '_evaluate', new=main_eval):
            err = run(wa._async_execute_action(0, 'foo', press_enter=True))
        assert '[0] is a <button>, not a text field' in err
        assert 'browse_act("0")' in err

    def test_iframe_element_types_via_main_tab_and_enters_via_frame(self):
        # OOPIF field: focus+clear in the frame session, Input.insertText on the
        # main tab (input routing follows focus), Enter through the frame session.
        wa = make_wa(_browse_snapshot_frame_ids=['ft1'])
        frame_eval = AsyncMock(side_effect=[META_TEXT, None, True, 'foo'])  # meta, scroll, focus, read-back
        with patch.object(WebAccess, '_async_eval_in_frame_target', new=frame_eval), \
                patch.object(WebAccess, '_evaluate', new=AsyncMock(return_value=True)) as main_eval, \
                patch.object(WebAccess, '_async_dispatch_key', new=AsyncMock()) as dispatch:
            assert run(wa._async_execute_action(0, 'foo', press_enter=True)) is None
        assert 'el.focus()' in frame_eval.await_args_list[2].args[1]
        assert "setter.call(el, value)" not in ''.join(c.args[1] for c in frame_eval.await_args_list)
        dispatch.assert_awaited_once_with('enter', 'ft1')
        main_eval.assert_awaited_once()  # the activeElement-is-an-iframe guard
        wa._tab.send.assert_awaited_once()  # the main-tab insertText


def key_patches(dispatch=None, focused=True):
    return (
        patch.object(WebAccess, '_async_browse_ready_midsession',
                     new=AsyncMock(return_value=None)),
        patch.object(WebAccess, '_async_dispatch_key', new=dispatch or AsyncMock()),
        patch.object(WebAccess, '_async_sync_popup_targets', new=AsyncMock()),
        patch.object(WebAccess, '_async_wait_for_page_ready', new=AsyncMock()),
        patch.object(WebAccess, '_async_take_snapshot',
                     new=AsyncMock(return_value='SNAP')),
        patch.object(WebAccess, '_evaluate', new=AsyncMock(return_value=focused)),
    )


@pytest.mark.unit
class TestBrowseKey:

    def _key(self, spec, wa=None, dispatch=None, focused=True):
        wa = wa or make_wa(_browse_snapshot_frame_ids=[None, None, None, None])
        dispatch = dispatch or AsyncMock()
        p = key_patches(dispatch, focused)
        with p[0], p[1], p[2], p[3], p[4], p[5]:
            result = run(wa._async_browse_key(spec))
        return result, dispatch

    def test_unknown_key_lists_supported_set(self):
        result, dispatch = self._key('F5')
        assert result.startswith("ERROR: unsupported key 'F5'")
        assert ('Enter, Escape, Tab, Backspace, ArrowUp, ArrowDown, '
                'ArrowLeft, ArrowRight') in result
        dispatch.assert_not_awaited()

    def test_bad_index_errors(self):
        result, dispatch = self._key('x,Enter')
        assert result == "ERROR: 'x' is not a valid index."
        dispatch.assert_not_awaited()

    def test_plain_key_presses_on_focused_element(self):
        result, dispatch = self._key('Enter')
        assert result == 'SNAP'
        dispatch.assert_awaited_once_with('Enter', None)

    def test_indexed_key_focuses_element_first(self):
        wa = make_wa(_browse_snapshot_frame_ids=[None, None, None, None])
        dispatch = AsyncMock()
        focus_eval = AsyncMock(return_value=True)
        p = key_patches(dispatch)
        with p[0], p[1], p[2], p[3], p[4], \
                patch.object(WebAccess, '_evaluate', new=focus_eval):
            result = run(wa._async_browse_key('3,ArrowDown'))
        assert result == 'SNAP'
        assert 'data-floxy-snap-idx="3"' in focus_eval.await_args.args[1]
        assert '.focus()' in focus_eval.await_args.args[1]
        dispatch.assert_awaited_once_with('ArrowDown', None)

    def test_indexed_key_on_iframe_element_uses_frame_eval(self):
        wa = make_wa(_browse_snapshot_frame_ids=[None, 'ft7'])
        dispatch = AsyncMock()
        frame_eval = AsyncMock(return_value=True)
        p = key_patches(dispatch)
        with p[0], p[1], p[2], p[3], p[4], p[5], \
                patch.object(WebAccess, '_async_eval_in_frame_target', new=frame_eval):
            result = run(wa._async_browse_key('1,Escape'))
        assert result == 'SNAP'
        assert frame_eval.await_args.args[0] == 'ft7'
        dispatch.assert_awaited_once_with('Escape', 'ft7')

    def test_out_of_range_index_returns_error_with_snapshot(self):
        result, dispatch = self._key('9,Enter')
        assert result.startswith('ERROR: index 9 is out of range')
        assert 'SNAP' in result
        dispatch.assert_not_awaited()

    def test_no_snapshot_with_index_errors(self):
        wa = make_wa(_browse_snapshot_frame_ids=[])
        result, dispatch = self._key('0,Enter', wa=wa)
        assert result == 'ERROR: no current snapshot. Call browse_open(url) again.'
        dispatch.assert_not_awaited()

    def test_vanished_element_returns_error_with_snapshot(self):
        result, dispatch = self._key('2,Enter', focused=False)
        assert result.startswith('ERROR: element [2] is no longer in the DOM')
        assert 'SNAP' in result
        dispatch.assert_not_awaited()

    def test_dispatch_failure_reports_error(self):
        dispatch = AsyncMock(side_effect=RuntimeError('transport gone'))
        result, _ = self._key('Enter', dispatch=dispatch)
        assert result == 'ERROR: key press failed: RuntimeError: transport gone'

    def test_midsession_gate_result_is_returned(self):
        wa = make_wa()
        with patch.object(WebAccess, '_async_browse_ready_midsession',
                          new=AsyncMock(return_value='NOTE: restarted')), \
                patch.object(WebAccess, '_async_dispatch_key', new=AsyncMock()) as dispatch:
            assert run(wa._async_browse_key('Enter')) == 'NOTE: restarted'
        dispatch.assert_not_awaited()
