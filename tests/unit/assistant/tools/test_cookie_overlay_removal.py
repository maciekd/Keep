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
        this.shadowRoot = null;
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
    click() {}
}

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
}));
"""


def run_consent_script(spec: dict) -> dict:
    script = WebAccess._COOKIE_REMOVAL_JS.strip().rstrip(';')
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

    def test_positioned_dialog_backdrop_is_removed(self):
        dialog = {'tag': 'div', 'id': 'dlg', 'cls': 'overlay', 'attrs': {'role': 'dialog'},
                  'style': {'position': 'absolute'}, 'rect': {'width': 400, 'height': 300}}
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
        assert out['result'] == {'clicked': 0, 'removed': 0, 'kept': 2}


@pytest.mark.unit
class TestConsentScriptGuards:
    """Tripwires that hold without node."""

    def test_no_substring_backdrop_selector_in_final_cleanup(self):
        js = WebAccess._COOKIE_REMOVAL_JS
        assert """querySelectorAll('[class*="backdrop"], [class*="overlay"]""" not in js
        assert "querySelectorAll('.backdrop, .overlay, .modal-backdrop')" in js

    def test_every_removal_goes_through_the_guard(self):
        js = WebAccess._COOKIE_REMOVAL_JS
        assert js.count('el.remove()') == 1
        assert 'if (isProtected(el)) { kept++; return false; }' in js


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

    def test_numeric_heights_still_scroll(self):
        wa = WebAccess.__new__(WebAccess)
        tab = Mock()
        tab.evaluate = AsyncMock()
        with patch.object(WebAccess, '_evaluate', new=AsyncMock(side_effect=[900, 800, 900, 900])), \
                patch('assistant.tools.WebAccess.asyncio.sleep', new=AsyncMock()):
            asyncio.run(wa._async_scroll_to_load_content(tab))
        scrolled = [c.args[0] for c in tab.evaluate.await_args_list]
        assert scrolled == ['window.scrollTo(0, 800);', 'window.scrollTo(0, 1600);', 'window.scrollTo(0, 0);']

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

    def test_json_without_readable_leaves_adds_no_header(self):
        payload = json.dumps({'image': 'https://shop.example/lamp.png', 'gift': False, 'meta': None})
        assert WebAccess._summarise_embedded_json([payload]) == ""
        text = WebAccess.extract_text_with_links(self.SHELL.format(payload=payload), 'https://shop.example/o')
        assert text == 'Order status'
