"""The consent script must never delete the document: Apple's JS classes <html> with
'supports-backdrop-filter', which the old substring selector took for a backdrop and
removed — every Apple Store page came back blank. The script runs here in node against
a minimal DOM stub; the scroll-to-load follow-on crash and the embedded-JSON reader
for such client-rendered pages are covered too."""

import asyncio
import json
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from assistant.tools.WebAccess import WebAccess

NODE = shutil.which('node')

# Just enough DOM for the consent script: a tree of elements with classes, attributes,
# computed style and a rect, and a selector matcher for comma lists of simple selectors
# (tag, #id, .class, [attr="v"], [attr*="v"]). Anything fancier matches nothing.
_DOM_STUB_JS = r"""
class El {
    constructor(spec, parent) {
        this.tagName = spec.tag.toUpperCase();
        this.id = spec.id || '';
        this.className = spec.cls || '';
        this.attrs = Object.assign({}, spec.attrs || {});
        if (this.id) this.attrs.id = this.id;
        if (this.className) this.attrs['class'] = this.className;
        this.style = {};
        this.computed = Object.assign({position: 'static', zIndex: 'auto'}, spec.style || {});
        this.rect = spec.rect || {width: 0, height: 0};
        this.ownText = spec.text || '';
        this.parentElement = parent || null;
        this.children = (spec.children || []).map(c => new El(c, this));
        this.offsetParent = parent || null;
        this.shadowRoot = spec.shadow ? new El(spec.shadow, null) : null;
    }
    get textContent() { return this.ownText + this.children.map(c => c.textContent).join(' '); }
    get innerText() { return this.textContent; }
    get offsetHeight() { return this.rect.height; }
    getAttribute(name) { return name in this.attrs ? this.attrs[name] : null; }
    getBoundingClientRect() { return this.rect; }
    descendants() { return this.children.flatMap(c => [c, ...c.descendants()]); }
    contains(other) { return other === this || this.descendants().includes(other); }
    matches(selector) {
        return selector.split(',').map(s => s.trim()).some(simple => {
            let m;
            if ((m = /^\[([\w-]+)(\*?)="([^"]*)"\]$/.exec(simple))) {
                const v = this.getAttribute(m[1]);
                return v !== null && (m[2] ? v.includes(m[3]) : v === m[3]);
            }
            if ((m = /^\.([\w-]+)$/.exec(simple))) return this.className.split(/\s+/).includes(m[1]);
            if ((m = /^#([\w-]+)$/.exec(simple))) return this.id === m[1];
            if ((m = /^[a-z]+$/.exec(simple))) return this.tagName.toLowerCase() === simple;
            return false;
        });
    }
    querySelectorAll(selector) { return this.descendants().filter(e => e.matches(selector)); }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    remove() {
        if (this.parentElement) {
            this.parentElement.children = this.parentElement.children.filter(c => c !== this);
        } else {
            document.documentElement = null;
        }
        this.parentElement = null;
    }
    click() { clicks.push(this.id || this.className); }
}
const clicks = [];

const html = new El(SPEC, null);
globalThis.document = {
    documentElement: html,
    get body() { return this.documentElement && this.documentElement.querySelector('body'); },
    get head() { return this.documentElement && this.documentElement.querySelector('head'); },
    querySelectorAll(selector) {
        const root = this.documentElement;
        return root ? [root, ...root.descendants()].filter(e => e.matches(selector)) : [];
    },
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; },
};
globalThis.window = {innerWidth: 1280, innerHeight: 800, getComputedStyle: el => el.computed};

const result = SCRIPT;
const root = document.documentElement;
console.log(JSON.stringify({
    result: result,
    ids: root ? [root, ...root.descendants()].map(e => e.id).filter(Boolean) : [],
    text: root ? root.textContent : '',
    clicks: clicks,
}));
"""


def run_consent_script(spec: dict, script: str = WebAccess._COOKIE_REMOVAL_JS) -> dict:
    script = script.strip().rstrip(';')
    program = _DOM_STUB_JS.replace('SPEC', json.dumps(spec)).replace('SCRIPT', script)
    done = subprocess.run([NODE, '-'], input=program, capture_output=True, text=True, timeout=20)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def page(html_cls: str, *body_children: dict, body_cls: str = '') -> dict:
    return {'tag': 'html', 'id': 'root', 'cls': html_cls, 'children': [
        {'tag': 'head', 'id': 'head'},
        {'tag': 'body', 'id': 'body', 'cls': body_cls, 'children': list(body_children)},
    ]}


ORDER = {'tag': 'main', 'id': 'order', 'text': 'Order KEP-47 — arrives Thursday'}
FULLSCREEN = {'width': 1280, 'height': 800}


@pytest.mark.unit
@pytest.mark.skipif(NODE is None, reason="node is needed to run the consent script")
class TestConsentScriptKeepsTheDocument:

    def test_html_with_backdrop_feature_class_survives_untouched(self):
        out = run_consent_script(page('js supports-backdrop-filter no-touch', ORDER))
        assert out['ids'] == ['root', 'head', 'body', 'order']
        assert 'arrives Thursday' in out['text']
        assert out['result'] == {'clicked': 0, 'removed': 0, 'kept': 0}

    def test_fixed_position_cookie_overlay_is_removed(self):
        # No text and a z-index below pass 3's threshold: only the backdrop cleanup can take it.
        backdrop = {'tag': 'div', 'id': 'veil', 'cls': 'modal-backdrop fade',
                    'style': {'position': 'fixed', 'zIndex': '50'}, 'rect': FULLSCREEN}
        out = run_consent_script(page('supports-backdrop-filter', ORDER, backdrop))
        assert out['ids'] == ['root', 'head', 'body', 'order']
        assert out['result']['removed'] == 1

    def test_cookie_banner_with_text_is_removed(self):
        banner = {'tag': 'div', 'id': 'cookie-bar', 'cls': 'overlay',
                  'style': {'position': 'fixed', 'zIndex': '9999'}, 'rect': {'width': 1280, 'height': 160},
                  'text': 'We use cookies', 'children': [
                      {'tag': 'button', 'id': 'ok', 'cls': 'cookie-accept', 'text': 'Accept all'}]}
        out = run_consent_script(page('', ORDER, banner))
        assert out['ids'] == ['root', 'head', 'body', 'order']
        assert out['result']['clicked'] == 1

    def test_overlay_class_in_normal_flow_is_kept(self):
        hero = {'tag': 'div', 'id': 'hero', 'cls': 'overlay', 'rect': FULLSCREEN, 'text': 'Autumn sale'}
        out = run_consent_script(page('', ORDER, hero))
        assert 'hero' in out['ids']
        assert out['result']['removed'] == 0

    def test_small_positioned_overlay_without_layer_or_role_is_kept(self):
        badge = {'tag': 'span', 'id': 'badge', 'cls': 'overlay',
                 'style': {'position': 'absolute', 'zIndex': '2'}, 'rect': {'width': 40, 'height': 20}}
        out = run_consent_script(page('', ORDER, badge))
        assert 'badge' in out['ids']

    def test_positioned_consent_dialog_backdrop_is_removed(self):
        dialog = {'tag': 'div', 'id': 'dlg', 'cls': 'overlay', 'attrs': {'role': 'dialog'},
                  'style': {'position': 'absolute'}, 'rect': {'width': 400, 'height': 300},
                  'text': 'Your privacy choices'}
        out = run_consent_script(page('', ORDER, dialog))
        assert 'dlg' not in out['ids']

    def test_body_matching_a_banner_selector_is_kept(self):
        out = run_consent_script(page('', {'tag': 'main', 'id': 'order', 'text': 'Accept the delivery by Friday'},
                                      body_cls='has-banner'))
        assert out['ids'] == ['root', 'head', 'body', 'order']
        assert out['result']['kept'] >= 1

    def test_wrapper_around_main_content_is_kept(self):
        wrapper = {'tag': 'div', 'id': 'shell', 'cls': 'overlay',
                   'style': {'position': 'fixed', 'zIndex': '5000'}, 'rect': FULLSCREEN, 'children': [ORDER]}
        out = run_consent_script(page('', wrapper))
        assert out['ids'] == ['root', 'head', 'body', 'shell', 'order']
        # only the backdrop cleanup reaches for it: pass 3 wants consent wording
        assert out['result'] == {'clicked': 0, 'removed': 0, 'kept': 1}

    def test_script_source_does_not_dilute_a_content_wrapper(self):
        # No <main>: the share of the page's text is all that protects the wrapper, and
        # the JSON blob beside it must not count as page text.
        wrapper = {'tag': 'div', 'id': 'shell', 'cls': 'overlay', 'text': 'Order KEP-47 arrives Thursday',
                   'style': {'position': 'fixed', 'zIndex': '5000'}, 'rect': FULLSCREEN}
        blob = {'tag': 'script', 'id': 'init_data', 'text': '{"orderDetail": "%s"}' % ('x' * 4000)}
        out = run_consent_script(page('', wrapper, blob))
        assert 'shell' in out['ids']
        assert out['result']['removed'] == 0

    def test_script_source_does_not_protect_a_consent_wall(self):
        wall = {'tag': 'div', 'id': 'wall', 'cls': 'overlay', 'text': 'We use cookies',
                'style': {'position': 'fixed', 'zIndex': '9999'}, 'rect': FULLSCREEN, 'children': [
                    {'tag': 'script', 'id': 'vendors', 'text': '{"vendors": "%s"}' % ('x' * 4000)}]}
        content = {'tag': 'div', 'id': 'content', 'text': 'The lamp ships on Thursday. ' * 10}
        out = run_consent_script(page('', content, wall))
        assert out['ids'] == ['root', 'head', 'body', 'content']


@pytest.mark.unit
@pytest.mark.skipif(NODE is None, reason="node is needed to run the consent script")
class TestConsentScriptKeepsTheSitesOwnDialog:
    """A shop's delivery picker (2026-11): the pass took a node of the open modal, the
    framework crashed on its next render and the purchase failed three times out of three."""

    RAISED = {'position': 'fixed', 'zIndex': '5000'}
    PICKER_TEXT = 'Choose a delivery slot — Thursday 14–16'
    # enough page that the picker is not most of its text — that alone protects an element,
    # and with the one-line ORDER these tests passed with the fix undone
    SHOP = {'tag': 'main', 'id': 'order', 'text': 'Order KEP-47 — arrives Thursday. ' + 'Oat milk, 1 l. ' * 40}

    def test_a_tall_raised_modal_without_consent_wording_is_kept(self):
        picker = {'tag': 'div', 'id': 'picker', 'cls': 'modal', 'style': self.RAISED,
                  'rect': {'width': 600, 'height': 500}, 'text': self.PICKER_TEXT}
        out = run_consent_script(page('', self.SHOP, picker))
        assert 'picker' in out['ids']
        assert out['result']['removed'] == 0

    @pytest.mark.parametrize('attrs', [{'role': 'dialog'}, {'role': 'alertdialog'},
                                       {'aria-modal': 'true'}, {'data-state': 'open'}])
    def test_an_open_dialog_classed_as_an_overlay_is_kept(self, attrs):
        # 'We accept cards' is no consent wording, though pass 3 takes "accept" from a plain overlay
        picker = {'tag': 'div', 'id': 'picker', 'cls': 'overlay', 'attrs': attrs, 'style': self.RAISED,
                  'rect': FULLSCREEN, 'text': 'We accept cards. ' + self.PICKER_TEXT}
        out = run_consent_script(page('', self.SHOP, picker))
        assert 'picker' in out['ids']
        assert out['result']['removed'] == 0

    def test_an_open_dialog_about_cookies_is_still_removed(self):
        wall = {'tag': 'div', 'id': 'wall', 'cls': 'overlay', 'attrs': {'role': 'dialog', 'data-state': 'open'},
                'style': self.RAISED, 'rect': FULLSCREEN, 'text': 'We use cookies'}
        out = run_consent_script(page('', ORDER, wall))
        assert 'wall' not in out['ids']


@pytest.mark.unit
@pytest.mark.skipif(NODE is None, reason="node is needed to run the consent script")
class TestClickOnlyConsentPass:
    """The pass a covered click runs presses accept buttons and takes nothing out."""

    BANNER = {'tag': 'div', 'id': 'cookie-bar', 'cls': 'overlay',
              'style': {'position': 'fixed', 'zIndex': '9999'}, 'rect': {'width': 1280, 'height': 160},
              'text': 'We use cookies', 'children': [
                  {'tag': 'button', 'id': 'ok', 'cls': 'cookie-accept', 'text': 'Accept all'}]}

    def test_the_accept_button_is_pressed_and_the_banner_left_to_the_page(self):
        out = run_consent_script(page('', ORDER, self.BANNER), WebAccess._CONSENT_CLICK_JS)
        assert out['clicks'] == ['ok']
        assert out['result'] == {'clicked': 1, 'removed': 0, 'kept': 0}
        assert 'cookie-bar' in out['ids']

    def test_what_the_full_pass_removes_stays(self):
        backdrop = {'tag': 'div', 'id': 'veil', 'cls': 'modal-backdrop',
                    'style': {'position': 'fixed', 'zIndex': '50'}, 'rect': FULLSCREEN}
        spec = page('', ORDER, backdrop)
        assert 'veil' not in run_consent_script(spec)['ids']
        out = run_consent_script(spec, WebAccess._CONSENT_CLICK_JS)
        assert 'veil' in out['ids']
        assert out['result'] == {'clicked': 0, 'removed': 0, 'kept': 0}


@pytest.mark.unit
@pytest.mark.skipif(NODE is None, reason="node is needed to run the consent script")
class TestConsentScriptAnswersUsercentricsCmpUi:
    """The newer Usercentrics layer (fedex.com, #82): an <aside> shadow host whose
    overlay caught every real click. Only a button the layer shows is clicked."""

    @staticmethod
    def wall(*buttons):
        return {'tag': 'aside', 'id': 'usercentrics-cmp-ui', 'shadow': {
            'tag': 'div', 'id': 'cmp-wrapper', 'children': [
                {'tag': 'div', 'id': 'uc-overlay', 'cls': 'overlay'}, *buttons]}}

    def test_accept_all_is_preferred_when_shown(self):
        accept = {'tag': 'button', 'id': 'accept', 'cls': 'uc-accept-button', 'rect': {'width': 200, 'height': 40}}
        save = {'tag': 'button', 'id': 'save', 'cls': 'uc-save-button', 'rect': {'width': 200, 'height': 40}}
        out = run_consent_script(page('', ORDER, self.wall(save, accept)))
        assert out['clicks'] == ['accept']
        assert out['result']['clicked'] == 1

    def test_hidden_accept_all_yields_to_confirm_selection(self):
        accept = {'tag': 'button', 'id': 'accept', 'cls': 'uc-accept-button'}
        save = {'tag': 'button', 'id': 'save', 'cls': 'uc-save-button', 'rect': {'width': 200, 'height': 40}}
        out = run_consent_script(page('', ORDER, self.wall(accept, save)))
        assert out['clicks'] == ['save']

    def test_a_layer_showing_no_button_is_left_alone(self):
        accept = {'tag': 'button', 'id': 'accept', 'cls': 'uc-accept-button'}
        out = run_consent_script(page('', ORDER, self.wall(accept)))
        assert out['clicks'] == []
        assert out['result']['clicked'] == 0
        assert 'order' in out['ids']


@pytest.mark.unit
class TestScrollToLoadEvalResult:

    def test_exception_details_result_skips_scrolling(self):
        wa = WebAccess.__new__(WebAccess)
        tab = Mock()
        tab.evaluate = AsyncMock()
        failure = SimpleNamespace(exception_id=1, text="Uncaught TypeError")
        with patch.object(WebAccess, '_evaluate', new=AsyncMock(return_value=failure)), \
                patch('assistant.tools.WebAccess.trace') as trace:
            asyncio.run(wa._async_scroll_to_load_content(tab))
        tab.evaluate.assert_not_awaited()
        assert 'skipped' in trace.call_args.args[0]

    @staticmethod
    def _scroll(heights):
        wa = WebAccess.__new__(WebAccess)
        tab = Mock()
        tab.evaluate = AsyncMock()
        with patch.object(WebAccess, '_evaluate', new=AsyncMock(side_effect=heights)), \
                patch('assistant.tools.WebAccess.asyncio.sleep', new=AsyncMock()):
            asyncio.run(wa._async_scroll_to_load_content(tab))
        return [c.args[0] for c in tab.evaluate.await_args_list]

    def test_static_page_gets_one_jump_to_the_bottom(self):
        # A page whose height does not move after the jump has nothing more to load:
        # one jump, back to the top — not a walk of every viewport.
        assert self._scroll([9000, 9000]) == ['window.scrollTo(0, 9000);', 'window.scrollTo(0, 0);']

    def test_growing_page_is_jumped_again_until_it_stops(self):
        assert self._scroll([900, 1800, 2400, 2400]) == [
            'window.scrollTo(0, 900);', 'window.scrollTo(0, 1800);', 'window.scrollTo(0, 2400);',
            'window.scrollTo(0, 0);']

    def test_endless_page_stops_at_the_round_cap(self):
        heights = [900 * (i + 1) for i in range(WebAccess._SCROLL_LOAD_MAX_ROUNDS + 5)]
        scrolled = self._scroll(heights)
        assert len(scrolled) == WebAccess._SCROLL_LOAD_MAX_ROUNDS + 1
        assert scrolled[-1] == 'window.scrollTo(0, 0);'

    @pytest.mark.parametrize("value, expected", [(0, 0), (12.5, 12.5), (True, None), (None, None), ("9", None)])
    def test_eval_number(self, value, expected):
        assert WebAccess._eval_number(value) == expected


@pytest.mark.unit
class TestEmbeddedJsonSummary:

    SHELL = ('<html class="supports-backdrop-filter"><body><main><h1>Order status</h1></main>'
             '<script type="application/json" id="init_data">{payload}</script></body></html>')

    def test_thin_page_gets_its_json_summarised(self):
        payload = json.dumps({'orderDetail': {'orderNumber': 'KEP-47', 'items': [
            {'productName': 'Desk lamp', 'status': 'Shipped', 'image': 'https://shop.example/lamp.png',
             'gift': False, 'quantity': 2}]}})
        text = WebAccess.extract_text_with_links(self.SHELL.format(payload=payload), 'https://shop.example/o')
        assert text.startswith('Order status')
        assert '[Data from embedded JSON]' in text
        assert 'orderDetail.orderNumber: KEP-47' in text
        assert 'items.status: Shipped' in text
        assert 'items.quantity: 2' in text
        assert 'lamp.png' not in text and 'gift' not in text

    def test_page_with_plenty_of_text_is_left_alone(self):
        html = ('<html><body><main>' + '<p>The lamp ships on Thursday.</p>' * 40 + '</main>'
                '<script type="application/json">{"status": "Shipped"}</script></body></html>')
        assert 'embedded JSON' not in WebAccess.extract_text_with_links(html, 'https://shop.example/o')

    def test_broken_json_is_ignored(self):
        text = WebAccess.extract_text_with_links(self.SHELL.format(payload='{oops'), 'https://shop.example/o')
        assert text == 'Order status'

    def test_summary_is_capped(self):
        block = json.dumps({f'k{i}': f'value number {i}' for i in range(1000)})
        summary = WebAccess._summarise_embedded_json([block])
        assert len(summary) < WebAccess._EMBEDDED_JSON_MAX_CHARS + 100

    def test_next_data_is_left_to_the_product_reader(self):
        # A thin Next.js page: __NEXT_DATA__ goes through the product summary only, never
        # dumped a second time as key/value lines (and its session data stays out).
        next_data = json.dumps({'props': {'pageProps': {'product': {
            'name': 'Cotton Tee', 'code': 'CT001', 'skus': [{'size': 'M', 'inStock': True}]},
            'session': {'token': 'sess-1234'}}}})
        html = (f'<html><body><main>Cotton Tee</main>'
                f'<script id="__NEXT_DATA__" type="application/json">{next_data}</script></body></html>')
        text = WebAccess.extract_text_with_links(html, 'https://shop.example/tee')
        assert '[Structured product data from __NEXT_DATA__]' in text
        assert '[Data from embedded JSON]' not in text
        assert 'sess-1234' not in text

    def test_only_readable_leaves_are_kept(self):
        block = json.dumps({
            'order': {'shipping': {'address': {'city': 'Uppsala', 'zip': 75236}}},
            'template': '<div>x</div>', 'route': '/orders/1', 'blob': '{"a":1}',
            'note': 'x' * (WebAccess._EMBEDDED_JSON_MAX_VALUE + 1), 'empty': '', 'missing': None,
            'count': 0,
        })
        summary = WebAccess._summarise_embedded_json([block])
        assert summary.splitlines() == ['[Data from embedded JSON]',
                                        'address.city: Uppsala', 'address.zip: 75236', 'count: 0']

    def test_every_parseable_block_contributes_once(self):
        items = [{'name': 'Desk lamp', 'status': 'Shipped'}] * 3
        blocks = ['{oops', json.dumps({'items': items}), json.dumps({'store': 'Downtown'})]
        summary = WebAccess._summarise_embedded_json(blocks)
        assert summary.splitlines() == ['[Data from embedded JSON]',
                                        'items.name: Desk lamp', 'items.status: Shipped', 'store: Downtown']

    def test_records_sharing_a_value_keep_all_their_fields(self):
        items = [{'name': 'Desk lamp', 'status': 'Shipped'}, {'name': 'Chair', 'status': 'Pending'},
                 {'name': 'Desk', 'status': 'Shipped'}]
        summary = WebAccess._summarise_embedded_json([json.dumps({'items': items})])
        assert summary.splitlines()[1:] == ['items.name: Desk lamp', 'items.status: Shipped',
                                            'items.name: Chair', 'items.status: Pending',
                                            'items.name: Desk', 'items.status: Shipped']

    def test_nested_records_stay_with_their_parent_and_repeat_blocks_collapse(self):
        items = [{'name': 'Desk', 'options': [{'name': 'Gift wrap'}, {'name': 'Gift wrap'}]},
                 {'name': 'Chair', 'options': [{'name': 'Gift wrap'}]}]
        block = json.dumps({'items': items})
        summary = WebAccess._summarise_embedded_json([block, block])
        assert summary.splitlines()[1:] == ['items.name: Desk', 'options.name: Gift wrap',
                                            'items.name: Chair', 'options.name: Gift wrap']

    def test_repeated_records_do_not_use_up_the_cap(self):
        items = [{'name': 'Desk lamp', 'note': 'n' * 150}] * 100 + [{'name': 'Chair'}]
        summary = WebAccess._summarise_embedded_json([json.dumps({'items': items})])
        assert summary.splitlines()[-1] == 'items.name: Chair'

    def test_top_level_scalars_have_no_empty_key(self):
        assert WebAccess._summarise_embedded_json(['[7, "Downtown"]']).splitlines()[1:] == ['7', 'Downtown']

    def test_json_without_readable_leaves_adds_no_header(self):
        payload = json.dumps({'image': 'https://shop.example/lamp.png', 'gift': False, 'meta': None})
        assert WebAccess._summarise_embedded_json([payload]) == ""
        text = WebAccess.extract_text_with_links(self.SHELL.format(payload=payload), 'https://shop.example/o')
        assert text == 'Order status'


@pytest.mark.unit
@pytest.mark.skipif(NODE is None, reason="node is needed to run the consent script")
class TestTheSitesDialogInsideAWrapper:
    """Modal libraries put the dialog one level down: a fixed overlay element with the
    role=dialog content as its child. Taking the wrapper takes the open dialog with it."""

    PICKER_TEXT = 'Choose a delivery slot — Thursday 14–16'
    # enough page that the dialog is not most of its text (which alone would protect it)
    SHOP = {'tag': 'main', 'id': 'order', 'text': 'Order KEP-47 — arrives Thursday. ' + 'Oat milk, 1 l. ' * 40}

    def wrapped(self, cls: str, style: dict, text: str) -> dict:
        return {'tag': 'div', 'id': 'wrapper', 'cls': cls, 'style': style, 'rect': FULLSCREEN, 'children': [
            {'tag': 'div', 'id': 'picker', 'attrs': {'role': 'dialog', 'aria-modal': 'true'},
             'rect': {'width': 600, 'height': 500}, 'text': text}]}

    @pytest.mark.parametrize('cls, style, text', [
        # the backdrop cleanup: a full-screen .overlay, no wording needed at all
        ('overlay', {'position': 'fixed', 'zIndex': '50'}, PICKER_TEXT),
        # pass 3: a raised wrapper whose dialog happens to say "accept"
        ('checkout-modal', {'position': 'fixed', 'zIndex': '5000'}, 'We accept cards. ' + PICKER_TEXT),
    ], ids=['backdrop-cleanup', 'pass-3'])
    def test_a_wrapper_holding_an_open_dialog_of_the_sites_own_is_kept(self, cls, style, text):
        out = run_consent_script(page('', self.SHOP, self.wrapped(cls, style, text)))
        assert 'picker' in out['ids']
        assert out['result']['removed'] == 0

    def test_a_wrapper_holding_a_consent_dialog_still_goes(self):
        out = run_consent_script(page('', self.SHOP, self.wrapped(
            'overlay', {'position': 'fixed', 'zIndex': '5000'}, 'We use cookies to remember your basket')))
        assert 'wrapper' not in out['ids'] and 'picker' not in out['ids']
        assert 'arrives Thursday' in out['text']


@pytest.mark.unit
@pytest.mark.skipif(NODE is None, reason="node is needed to run the consent script")
class TestTheScriptAPassIsRunWith:
    """`click_only` is the whole difference between the pass at page load and the one under
    a covered click: what the method hands the page is run here, not compared by name."""

    VEIL = {'tag': 'div', 'id': 'veil', 'cls': 'modal-backdrop',
            'style': {'position': 'fixed', 'zIndex': '50'}, 'rect': FULLSCREEN}

    def ids_left_by(self, **kwargs) -> list:
        evaluate = AsyncMock(return_value={})
        with patch.object(WebAccess, '_evaluate', new=evaluate):
            asyncio.run(WebAccess._async_remove_cookies_banner(
                WebAccess.__new__(WebAccess), Mock(), 'https://shop.example/checkout', **kwargs))
        return run_consent_script(page('', ORDER, self.VEIL), evaluate.await_args.args[1])['ids']

    def test_the_click_only_pass_hands_the_page_a_script_that_removes_nothing(self):
        assert 'veil' in self.ids_left_by(click_only=True)

    def test_the_default_pass_still_removes(self):
        assert 'veil' not in self.ids_left_by()
