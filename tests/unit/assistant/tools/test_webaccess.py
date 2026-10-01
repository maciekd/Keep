import asyncio
import base64
import os
import signal
import time
import requests
import pytest
from unittest.mock import patch, Mock, AsyncMock
from assistant.tools.WebAccess import WebAccess


@pytest.mark.unit
class TestWebAccess:
    """Tests for WebAccess link preservation logic (static methods only, no Chrome needed)"""

    def test_extract_text_with_links_preserves_links(self):
        """Links should be preserved in <a href='...'>text</a> format"""
        html = """
        <html>
            <body>
                <p>Check out <a href="https://example.com">this site</a> for more info.</p>
            </body>
        </html>
        """
        result = WebAccess.extract_text_with_links(html, "https://test.com")

        assert '<a href="https://example.com">this site</a>' in result
        assert "Check out" in result
        assert "for more info." in result

    def test_extract_text_with_links_converts_relative_urls(self):
        """Relative URLs should be converted to absolute URLs"""
        html = """
        <html>
            <body>
                <a href="/about">About</a>
                <a href="contact.html">Contact</a>
            </body>
        </html>
        """
        result = WebAccess.extract_text_with_links(html, "https://example.com/page/")

        assert '<a href="https://example.com/about">About</a>' in result
        assert '<a href="https://example.com/page/contact.html">Contact</a>' in result

    def test_extract_text_with_links_removes_navigation(self):
        """Navigation, header, footer elements should be removed"""
        html = """
        <html>
            <body>
                <nav>
                    <a href="/nav-link">Nav Link</a>
                </nav>
                <header>
                    <a href="/header-link">Header Link</a>
                </header>
                <main>
                    <a href="/content-link">Content Link</a>
                </main>
                <footer>
                    <a href="/footer-link">Footer Link</a>
                </footer>
            </body>
        </html>
        """
        result = WebAccess.extract_text_with_links(html, "https://example.com")

        # Content link should be present
        assert "Content Link" in result
        assert '<a href="https://example.com/content-link">Content Link</a>' in result

        # Navigation, header, footer links should be removed
        assert "Nav Link" not in result
        assert "Header Link" not in result
        assert "Footer Link" not in result

    def test_extract_text_with_links_removes_scripts_and_styles(self):
        """Script and style tags should be completely removed"""
        html = """
        <html>
            <head>
                <style>body { color: red; }</style>
            </head>
            <body>
                <p>Visible text</p>
                <script>alert('hello');</script>
            </body>
        </html>
        """
        result = WebAccess.extract_text_with_links(html, "https://example.com")

        assert "Visible text" in result
        assert "color: red" not in result
        assert "alert" not in result

    def test_extract_text_with_links_handles_multiple_links(self):
        """Should handle pages with many links (like news sites)"""
        html = """
        <html>
            <body>
                <article>
                    <h2><a href="/article1">Article 1</a></h2>
                    <p>Description 1</p>
                </article>
                <article>
                    <h2><a href="/article2">Article 2</a></h2>
                    <p>Description 2</p>
                </article>
                <article>
                    <h2><a href="/article3">Article 3</a></h2>
                    <p>Description 3</p>
                </article>
            </body>
        </html>
        """
        result = WebAccess.extract_text_with_links(html, "https://news.com")

        # All articles should be present (not just the first one)
        assert "Article 1" in result
        assert "Article 2" in result
        assert "Article 3" in result
        assert "Description 1" in result
        assert "Description 2" in result
        assert "Description 3" in result

        # All links should be preserved
        assert result.count('<a href=') == 3

    def test_extract_text_with_links_skips_empty_links(self):
        """Links without text should be skipped"""
        html = """
        <html>
            <body>
                <a href="/page1">Valid Link</a>
                <a href="/page2"></a>
                <a href="/page3">   </a>
                <a href="/page4">Another Valid</a>
            </body>
        </html>
        """
        result = WebAccess.extract_text_with_links(html, "https://example.com")

        assert "Valid Link" in result
        assert "Another Valid" in result
        # Only 2 links should be preserved (the ones with text)
        assert result.count('<a href=') == 2

    def test_extract_text_falls_back_to_body_when_main_is_small(self):
        """When <main> holds less than 2/3 of body text, fall back to <body>"""
        html = """
        <html>
            <body>
                <main><p>Small main</p></main>
                <div><p>Important flight table data that lives outside main and is much longer than the main content itself</p></div>
            </body>
        </html>
        """
        result = WebAccess.extract_text_with_links(html, "https://example.com")
        assert "flight table data" in result

    def test_extract_text_uses_main_when_it_has_most_content(self):
        """When <main> holds the bulk of content, use it instead of <body>"""
        html = """
        <html>
            <body>
                <main><p>This is the main content area with plenty of text that dominates the page</p></main>
                <div><p>Tiny sidebar</p></div>
            </body>
        </html>
        """
        result = WebAccess.extract_text_with_links(html, "https://example.com")
        assert "main content area" in result
        assert "Tiny sidebar" not in result

    def test_extract_text_with_links_handles_nested_elements(self):
        """Should handle nested elements correctly"""
        html = """
        <html>
            <body>
                <div>
                    <p>Some text with <a href="/link1">a link</a> inside.</p>
                    <p>More text with <a href="/link2">another link</a> here.</p>
                </div>
            </body>
        </html>
        """
        result = WebAccess.extract_text_with_links(html, "https://example.com")

        assert "Some text with" in result
        assert '<a href="https://example.com/link1">a link</a>' in result
        assert "inside." in result
        assert "More text with" in result
        assert '<a href="https://example.com/link2">another link</a>' in result

    def test_extract_text_with_links_strips_excessive_whitespace(self):
        """Should clean up excessive whitespace"""
        html = """
        <html>
            <body>
                <p>Text    with    spaces</p>


                <p>Another paragraph</p>
            </body>
        </html>
        """
        result = WebAccess.extract_text_with_links(html, "https://example.com")

        # Should have content on separate lines, but no excessive blank lines
        lines = result.split('\n')
        assert "Text with spaces" in result or "Text    with    spaces" in result
        assert "Another paragraph" in result
        # No empty lines should remain
        assert all(line.strip() for line in lines)

    def test_extract_text_with_links_separates_table_cells(self):
        """Table elements should be treated as block-level, separating cells with newlines"""
        html = """
        <html>
            <body>
                <table>
                    <tr><th>Flight</th><th>Status</th></tr>
                    <tr><td>SK123</td><td>On time</td></tr>
                    <tr><td>LH456</td><td>Delayed</td></tr>
                </table>
            </body>
        </html>
        """
        result = WebAccess.extract_text_with_links(html, "https://example.com")

        lines = [l.strip() for l in result.strip().splitlines() if l.strip()]
        assert "Flight" in lines
        assert "SK123" in lines
        assert "On time" in lines
        assert "LH456" in lines
        assert "Delayed" in lines

    def test_extract_text_with_links_handles_special_characters(self):
        """Should handle special characters in URLs and text"""
        html = """
        <html>
            <body>
                <a href="/page?foo=bar&baz=qux">Link with params</a>
                <a href="/path/to/file.html#section">Link with anchor</a>
                <p>Text with "quotes" and 'apostrophes'</p>
            </body>
        </html>
        """
        result = WebAccess.extract_text_with_links(html, "https://example.com")

        assert "Link with params" in result
        assert "Link with anchor" in result
        assert "foo=bar" in result
        assert "#section" in result
        assert "quotes" in result
        assert "apostrophes" in result


@pytest.mark.unit
class TestCookieBannerRemoval:
    """Tests for cookie consent container removal in extract_text_with_links"""

    def test_removes_cookiebot_elements_by_id(self):
        html = """
        <html><body>
            <div id="CookieBanner"><p>This website uses cookies</p><button>Accept</button></div>
            <div id="CookieBannerOverlay"></div>
            <div><p>Real page content here.</p></div>
        </body></html>
        """
        result = WebAccess.extract_text_with_links(html, "https://example.com")
        assert "Real page content" in result
        assert "uses cookies" not in result
        assert "Accept" not in result

    def test_removes_onetrust_elements(self):
        html = """
        <html><body>
            <div id="onetrust-consent-sdk"><p>Cookie preferences</p></div>
            <div id="onetrust-banner-sdk"><p>We use cookies</p></div>
            <div><p>Actual content.</p></div>
        </body></html>
        """
        result = WebAccess.extract_text_with_links(html, "https://example.com")
        assert "Actual content" in result
        assert "Cookie preferences" not in result
        assert "We use cookies" not in result

    def test_removes_gdpr_banners(self):
        html = """
        <html><body>
            <div id="gdpr-consent-banner"><p>GDPR notice text</p></div>
            <div><p>Main content.</p></div>
        </body></html>
        """
        result = WebAccess.extract_text_with_links(html, "https://example.com")
        assert "Main content" in result
        assert "GDPR notice" not in result

    def test_removes_detailed_cookiebot_tables(self):
        html = """
        <html><body>
            <div id="CybotCookiebotDialogDetailTableNecessary">
                <table><tr><td>bugsnag-anonymous-id</td><td>Persistent</td></tr></table>
            </div>
            <div id="CookieBannerDetails"><p>Cookie categories and descriptions</p></div>
            <div><p>Page content.</p></div>
        </body></html>
        """
        result = WebAccess.extract_text_with_links(html, "https://example.com")
        assert "Page content" in result
        assert "bugsnag" not in result
        assert "Cookie categories" not in result

    def test_preserves_non_cookie_content(self):
        html = """
        <html><body>
            <div id="main-content"><h1>Welcome</h1><p>Important information</p></div>
            <div id="sidebar"><p>Related links</p></div>
        </body></html>
        """
        result = WebAccess.extract_text_with_links(html, "https://example.com")
        assert "Welcome" in result
        assert "Important information" in result


@pytest.mark.unit
class TestSearch:
    @patch('assistant.tools.WebAccess.ConfigManager')
    def test_missing_api_key_reports_the_key(self, mock_cm_cls):
        mock_cm = Mock()
        mock_cm.get_api_key.side_effect = KeyError("serper")
        mock_cm_cls.return_value = mock_cm

        wa = WebAccess.__new__(WebAccess)
        result = wa.search("test query")
        assert result.startswith("ERROR")
        assert "api_keys.serper" in result

    @patch('assistant.tools.WebAccess.ConfigManager')
    def test_placeholder_api_key_reports_the_key(self, mock_cm_cls):
        mock_cm = Mock()
        mock_cm.get_api_key.return_value = 'YOUR_SERPER_API_KEY_HERE'
        mock_cm_cls.return_value = mock_cm

        wa = WebAccess.__new__(WebAccess)
        result = wa.search("test query")
        assert result.startswith("ERROR")
        assert "api_keys.serper" in result

    @patch('assistant.tools.WebAccess.requests.post')
    @patch('assistant.tools.WebAccess.ConfigManager')
    def test_successful_response(self, mock_cm_cls, mock_post):
        mock_cm = Mock()
        mock_cm.get_api_key.return_value = 'real-key'
        mock_cm.get.side_effect = lambda k, d=None: {'search.country_hint': 'se', 'search.interface_language': 'en', 'search.results_limit': 6}.get(k, d)
        mock_cm_cls.return_value = mock_cm

        mock_response = Mock()
        mock_response.json.return_value = {
            'organic': [
                {'title': 'Result 1', 'link': 'https://r1.com', 'snippet': 'Snippet 1'},
                {'title': 'Result 2', 'link': 'https://r2.com', 'snippet': 'Snippet 2'},
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        wa = WebAccess.__new__(WebAccess)
        result = wa.search("test query")
        assert "Result 1" in result
        assert "https://r1.com" in result
        assert "Snippet 1" in result
        assert "Result 2" in result

    @patch('assistant.tools.WebAccess.requests.post')
    @patch('assistant.tools.WebAccess.ConfigManager')
    def test_empty_results(self, mock_cm_cls, mock_post):
        mock_cm = Mock()
        mock_cm.get_api_key.return_value = 'real-key'
        mock_cm.get.side_effect = lambda k, d=None: {'search.country_hint': 'se', 'search.interface_language': 'en', 'search.results_limit': 6}.get(k, d)
        mock_cm_cls.return_value = mock_cm

        mock_response = Mock()
        mock_response.json.return_value = {'organic': []}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        wa = WebAccess.__new__(WebAccess)
        result = wa.search("test query")
        assert result == "No search results found for: test query"

    @patch('assistant.tools.WebAccess.requests.post')
    @patch('assistant.tools.WebAccess.ConfigManager')
    def test_request_exception_returns_error(self, mock_cm_cls, mock_post):
        import requests as req
        mock_cm = Mock()
        mock_cm.get_api_key.return_value = 'real-key'
        mock_cm.get.side_effect = lambda k, d=None: {'search.country_hint': 'se', 'search.interface_language': 'en', 'search.results_limit': 6}.get(k, d)
        mock_cm_cls.return_value = mock_cm

        mock_post.side_effect = req.exceptions.ConnectionError("connection refused")

        wa = WebAccess.__new__(WebAccess)
        result = wa.search("test query")
        assert result.startswith("Error calling Serper API")


@pytest.mark.unit
class TestIsBlockedResponse:
    def test_none_is_blocked(self):
        assert WebAccess._is_blocked_response(None) is True

    def test_short_content_is_blocked(self):
        assert WebAccess._is_blocked_response("short") is True

    def test_cloudflare_challenge_is_blocked(self):
        html = "<html><head><title>Just a moment...</title></head><body>cf_chl_opt</body></html>"
        assert WebAccess._is_blocked_response(html) is True

    def test_cloudflare_verification_is_blocked(self):
        html = "<html><body><div id='cf-browser-verification'>Checking</div></body></html>"
        assert WebAccess._is_blocked_response(html) is True

    def test_datadome_is_blocked(self):
        html = "<html><body><script src='https://js.datadome.co/dd.js'></script>Content</body></html>"
        assert WebAccess._is_blocked_response(html) is True

    def test_access_denied_is_blocked(self):
        html = "<html><head><title>Access Denied</title></head><body>You have been blocked.</body></html>"
        assert WebAccess._is_blocked_response(html) is True

    def test_normal_page_is_not_blocked(self):
        html = "<html><body><p>This is a normal web page with real content that should not be blocked.</p></body></html>"
        assert WebAccess._is_blocked_response(html) is False

@pytest.mark.unit
class TestFetchViaCloudscraper:
    @patch('assistant.tools.WebAccess.cloudscraper')
    def test_success(self, mock_cs):
        mock_scraper = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>" + "Content here. " * 20 + "</body></html>"
        mock_scraper.get.return_value = mock_response
        mock_cs.create_scraper.return_value = mock_scraper

        wa = WebAccess.__new__(WebAccess)
        result = wa._fetch_via_cloudscraper("https://example.com")
        assert result is not None
        assert "Content here" in result

    @patch('assistant.tools.WebAccess.cloudscraper')
    def test_non_200_returns_none(self, mock_cs):
        mock_scraper = Mock()
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"
        mock_scraper.get.return_value = mock_response
        mock_cs.create_scraper.return_value = mock_scraper

        wa = WebAccess.__new__(WebAccess)
        assert wa._fetch_via_cloudscraper("https://example.com") is None

    @patch('assistant.tools.WebAccess.cloudscraper')
    def test_exception_returns_none(self, mock_cs):
        mock_cs.create_scraper.side_effect = Exception("network error")

        wa = WebAccess.__new__(WebAccess)
        assert wa._fetch_via_cloudscraper("https://example.com") is None

    @patch('assistant.tools.WebAccess.cloudscraper')
    def test_short_response_returns_none(self, mock_cs):
        mock_scraper = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "tiny"
        mock_scraper.get.return_value = mock_response
        mock_cs.create_scraper.return_value = mock_scraper

        wa = WebAccess.__new__(WebAccess)
        assert wa._fetch_via_cloudscraper("https://example.com") is None


@pytest.mark.unit
class TestFetchViaCurlCffi:
    @patch('assistant.tools.WebAccess.cffi_requests')
    def test_success(self, mock_cffi):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>" + "Real content here. " * 20 + "</body></html>"
        mock_cffi.get.return_value = mock_response

        wa = WebAccess.__new__(WebAccess)
        result = wa._fetch_via_curl_cffi("https://example.com")
        assert result is not None
        assert "Real content here" in result
        call_kwargs = mock_cffi.get.call_args
        assert call_kwargs[1]["impersonate"] == "chrome131"
        assert call_kwargs[1]["timeout"] == 20
        assert "Sec-Fetch-Site" in call_kwargs[1]["headers"]

    @patch('assistant.tools.WebAccess.cffi_requests')
    def test_non_200_returns_none(self, mock_cffi):
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"
        mock_cffi.get.return_value = mock_response

        wa = WebAccess.__new__(WebAccess)
        assert wa._fetch_via_curl_cffi("https://example.com") is None

    @patch('assistant.tools.WebAccess.cffi_requests')
    def test_exception_returns_none(self, mock_cffi):
        mock_cffi.get.side_effect = Exception("connection failed")

        wa = WebAccess.__new__(WebAccess)
        assert wa._fetch_via_curl_cffi("https://example.com") is None

    @patch('assistant.tools.WebAccess.cffi_requests')
    def test_short_response_returns_none(self, mock_cffi):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "tiny"
        mock_cffi.get.return_value = mock_response

        wa = WebAccess.__new__(WebAccess)
        assert wa._fetch_via_curl_cffi("https://example.com") is None


@pytest.mark.unit
class TestFetchWebpageFallback:
    """The 3-tier cascade: nodriver -> cloudscraper -> curl_cffi.

    These tests patch the per-tier methods directly so the loop thread / event
    loop machinery is irrelevant — the cascade logic in `fetch_webpage` is what's
    under test.
    """

    def _make_wa(self):
        wa = WebAccess.__new__(WebAccess)
        wa._browser = None
        wa._tab = None
        wa._driver_ready = Mock()
        wa._driver_ready.wait.return_value = True
        wa._loop_thread = Mock()
        # Default: tabbed-content / iframe-text submits return None and close any
        # coroutine they were handed so we don't get "coroutine never awaited"
        # warnings from these cascade tests.
        def _submit(coro, timeout=None):
            try:
                coro.close()
            except Exception:
                pass
            return None
        wa._loop_thread.submit = Mock(side_effect=_submit)
        return wa

    @patch.object(WebAccess, '_fetch_via_curl_cffi')
    @patch.object(WebAccess, '_fetch_via_cloudscraper')
    @patch.object(WebAccess, '_fetch_via_chrome')
    def test_chrome_success_skips_fallbacks(self, mock_chrome, mock_cs, mock_cffi):
        wa = self._make_wa()
        mock_chrome.return_value = "<html><body><p>Good content from nodriver that is long enough.</p></body></html>"

        result = wa.fetch_webpage("https://example.com")
        assert "Good content from nodriver" in result
        mock_cs.assert_not_called()
        mock_cffi.assert_not_called()

    @patch.object(WebAccess, '_fetch_via_curl_cffi')
    @patch.object(WebAccess, '_fetch_via_cloudscraper')
    @patch.object(WebAccess, '_fetch_via_chrome')
    def test_blocked_chrome_falls_back_to_cloudscraper(self, mock_chrome, mock_cs, mock_cffi):
        wa = self._make_wa()
        mock_chrome.return_value = "<html><head><title>Just a moment...</title></head><body>cf_chl_opt</body></html>"
        mock_cs.return_value = "<html><body><p>Real content from cloudscraper that is meaningful.</p></body></html>"

        result = wa.fetch_webpage("https://example.com")
        assert "Real content from cloudscraper" in result
        mock_cffi.assert_not_called()

    @patch.object(WebAccess, '_fetch_via_curl_cffi')
    @patch.object(WebAccess, '_fetch_via_cloudscraper')
    @patch.object(WebAccess, '_fetch_via_chrome')
    def test_both_blocked_falls_back_to_curl_cffi(self, mock_chrome, mock_cs, mock_cffi):
        wa = self._make_wa()
        mock_chrome.return_value = "<html><head><title>Access Denied</title></head><body></body></html>"
        mock_cs.return_value = None
        mock_cffi.return_value = "<html><body><p>Real content from curl_cffi that bypasses TLS fingerprinting.</p></body></html>"

        result = wa.fetch_webpage("https://example.com")
        assert "Real content from curl_cffi" in result

    @patch.object(WebAccess, '_fetch_via_curl_cffi')
    @patch.object(WebAccess, '_fetch_via_cloudscraper')
    @patch.object(WebAccess, '_fetch_via_chrome')
    def test_all_tiers_fail_returns_error(self, mock_chrome, mock_cs, mock_cffi):
        wa = self._make_wa()
        mock_chrome.return_value = None
        mock_cs.return_value = None
        mock_cffi.return_value = None

        result = wa.fetch_webpage("https://example.com")
        assert "ERROR" in result

    @patch.object(WebAccess, '_fetch_via_curl_cffi')
    @patch.object(WebAccess, '_fetch_via_cloudscraper')
    @patch.object(WebAccess, '_fetch_via_chrome')
    def test_url_normalization(self, mock_chrome, mock_cs, mock_cffi):
        wa = self._make_wa()
        mock_chrome.return_value = "<html><body><p>Content from normalized URL request here.</p></body></html>"

        wa.fetch_webpage("example.com")
        mock_chrome.assert_called_with("https://example.com")

    @patch.object(WebAccess, '_fetch_via_curl_cffi')
    @patch.object(WebAccess, '_fetch_via_cloudscraper')
    @patch.object(WebAccess, '_fetch_via_chrome')
    def test_protocol_relative_url(self, mock_chrome, mock_cs, mock_cffi):
        wa = self._make_wa()
        mock_chrome.return_value = "<html><body><p>Content from protocol relative URL.</p></body></html>"

        wa.fetch_webpage("//example.com/path")
        mock_chrome.assert_called_with("https://example.com/path")


@pytest.mark.unit
class TestFindChromeBinary:
    @patch('assistant.tools.WebAccess.shutil.which', return_value=None)
    @patch('assistant.tools.WebAccess.os.access', return_value=False)
    @patch('assistant.tools.WebAccess.os.path.isfile', return_value=False)
    def test_returns_none_when_not_found(self, mock_isfile, mock_access, mock_which):
        result = WebAccess._find_chrome_binary()
        assert result is None

    @patch('assistant.tools.WebAccess.shutil.which', side_effect=lambda n: "/usr/bin/chromium" if n == "chromium" else None)
    @patch('assistant.tools.WebAccess.os.access', return_value=False)
    @patch('assistant.tools.WebAccess.os.path.isfile', return_value=False)
    def test_falls_back_to_which(self, mock_isfile, mock_access, mock_which):
        result = WebAccess._find_chrome_binary()
        assert result == "/usr/bin/chromium"

    @patch('assistant.tools.WebAccess.os.access', side_effect=lambda p, _: p.endswith("Chromium"))
    @patch('assistant.tools.WebAccess.os.path.isfile', side_effect=lambda p: p.endswith("Chromium"))
    def test_skips_missing_prefers_first_found(self, mock_isfile, mock_access):
        result = WebAccess._find_chrome_binary()
        assert result == "/Applications/Chromium.app/Contents/MacOS/Chromium"


@pytest.mark.unit
class TestChromePidTracking:
    """PID tracking + cleanup. nodriver exposes the Chrome subprocess as `browser._process`."""

    def _make_wa(self):
        wa = WebAccess.__new__(WebAccess)
        wa._browser = None
        wa._tab = None
        wa._chrome_pids = set()
        wa._driver_ready = Mock()
        return wa

    def test_track_chrome_pids_records_process_pid(self):
        wa = self._make_wa()
        browser = Mock()
        proc = Mock()
        proc.pid = 12345
        browser._process = proc
        wa._browser = browser

        wa._track_chrome_pids()

        assert 12345 in wa._chrome_pids

    def test_track_chrome_pids_handles_missing_attrs(self):
        wa = self._make_wa()
        wa._browser = Mock(spec=[])  # no _process attribute

        wa._track_chrome_pids()

        assert wa._chrome_pids == set()

    def test_track_chrome_pids_records_nodriver_process_pid(self):
        wa = self._make_wa()
        browser = Mock(spec=['_process_pid'])
        browser._process_pid = 4242
        wa._browser = browser

        wa._track_chrome_pids()

        assert 4242 in wa._chrome_pids

    @patch.object(WebAccess, '_chrome_child_processes', staticmethod(lambda pid: []))
    @patch('assistant.tools.WebAccess.os.kill')
    def test_kill_tracked_chrome_pids_kills_all(self, mock_kill):
        wa = self._make_wa()
        wa._chrome_pids = {111, 222}

        wa._kill_tracked_chrome_pids()

        assert mock_kill.call_count == 2
        killed_pids = {call.args[0] for call in mock_kill.call_args_list}
        assert killed_pids == {111, 222}
        assert wa._chrome_pids == set()

    @patch('assistant.tools.WebAccess.os.kill')
    def test_kill_tracked_chrome_pids_kills_children_first(self, mock_kill):
        wa = self._make_wa()
        wa._chrome_pids = {100}
        child = Mock()
        child.pid = 101
        with patch.object(WebAccess, '_chrome_child_processes',
                          staticmethod(lambda pid: [child] if pid == 100 else [])):
            wa._kill_tracked_chrome_pids()

        child.kill.assert_called_once()
        mock_kill.assert_called_once_with(100, signal.SIGKILL)
        assert wa._chrome_pids == set()

    def test_chrome_child_processes_empty_for_missing_pid(self):
        import psutil as psutil_mod
        with patch('assistant.tools.WebAccess.psutil.Process', side_effect=psutil_mod.NoSuchProcess(999)):
            assert WebAccess._chrome_child_processes(999) == []

    @patch.object(WebAccess, '_chrome_child_processes', staticmethod(lambda pid: []))
    @patch('assistant.tools.WebAccess.os.kill', side_effect=ProcessLookupError)
    def test_kill_tracked_pids_ignores_already_dead(self, mock_kill):
        wa = self._make_wa()
        wa._chrome_pids = {111}

        wa._kill_tracked_chrome_pids()  # should not raise

        assert wa._chrome_pids == set()

    def test_shutdown_driver_stops_browser_and_kills_pids(self):
        wa = self._make_wa()
        browser = Mock(spec=['_process', 'stop'])
        proc = Mock()
        proc.pid = 777
        browser._process = proc
        wa._browser = browser
        wa._loop_thread = Mock()
        wa._loop_thread.is_running.return_value = False

        with patch('assistant.tools.WebAccess.os.kill') as mock_kill, \
                patch.object(WebAccess, '_chrome_child_processes', staticmethod(lambda pid: [])), \
                patch.object(wa, '_kill_orphan_chrome_using_profile') as mock_sweep:
            wa._shutdown_driver()

        browser.stop.assert_called_once()
        assert wa._browser is None
        mock_kill.assert_called_once_with(777, signal.SIGKILL)
        mock_sweep.assert_called_once()

    def test_shutdown_driver_swallows_browser_stop_exception(self):
        wa = self._make_wa()
        browser = Mock(spec=['_process', 'stop'])
        browser.stop.side_effect = Exception("session dead")
        proc = Mock()
        proc.pid = 888
        browser._process = proc
        wa._browser = browser
        wa._loop_thread = Mock()
        wa._loop_thread.is_running.return_value = False

        with patch('assistant.tools.WebAccess.os.kill') as mock_kill, \
                patch.object(WebAccess, '_chrome_child_processes', staticmethod(lambda pid: [])), \
                patch.object(wa, '_kill_orphan_chrome_using_profile'):
            wa._shutdown_driver()  # should not raise

        assert wa._browser is None
        mock_kill.assert_called_once_with(888, signal.SIGKILL)

    def test_reset_instance_calls_shutdown(self):
        wa = self._make_wa()
        wa._browser = Mock()
        wa._chrome_pids = set()
        WebAccess._instance = wa

        with patch.object(wa, '_shutdown_driver') as mock_shutdown:
            WebAccess.reset_instance()

        mock_shutdown.assert_called_once()
        assert WebAccess._instance is None


@pytest.mark.unit
class TestOrphanChromeSweep:
    """Pre-launch sweep that kills Chrome processes from prior server runs still holding our profile."""

    def _make_wa(self, profile_dir="/tmp/floxy_test_profile"):
        wa = WebAccess.__new__(WebAccess)
        wa._profile_dir = profile_dir
        return wa

    def _proc(self, pid, cmdline):
        proc = Mock()
        proc.pid = pid
        proc.info = {'pid': pid, 'cmdline': cmdline}
        return proc

    def test_terminates_chrome_using_our_profile(self):
        wa = self._make_wa("/tmp/p")
        match = self._proc(111, ['/usr/bin/chrome', '--headless', '--user-data-dir=/tmp/p'])
        other = self._proc(222, ['/usr/bin/chrome', '--user-data-dir=/tmp/different'])

        with patch('assistant.tools.WebAccess.psutil.process_iter', return_value=[match, other]), \
             patch('assistant.tools.WebAccess.psutil.wait_procs', return_value=([match], [])):
            wa._kill_orphan_chrome_using_profile()

        match.terminate.assert_called_once()
        other.terminate.assert_not_called()

    def test_skips_own_pid(self):
        wa = self._make_wa("/tmp/p")
        own = self._proc(os.getpid(), ['python', '--user-data-dir=/tmp/p'])

        with patch('assistant.tools.WebAccess.psutil.process_iter', return_value=[own]), \
             patch('assistant.tools.WebAccess.psutil.wait_procs', return_value=([], [])) as wait_mock:
            wa._kill_orphan_chrome_using_profile()

        own.terminate.assert_not_called()
        wait_mock.assert_not_called()  # no victims -> early return

    def test_sigkills_processes_that_ignore_sigterm(self):
        wa = self._make_wa("/tmp/p")
        stubborn = self._proc(333, ['/usr/bin/chrome', '--user-data-dir=/tmp/p'])

        with patch('assistant.tools.WebAccess.psutil.process_iter', return_value=[stubborn]), \
             patch('assistant.tools.WebAccess.psutil.wait_procs', return_value=([], [stubborn])):
            wa._kill_orphan_chrome_using_profile()

        stubborn.terminate.assert_called_once()
        stubborn.kill.assert_called_once()

    def test_no_orphans_is_noop(self):
        wa = self._make_wa("/tmp/p")
        unrelated = self._proc(444, ['/bin/sleep', '60'])

        with patch('assistant.tools.WebAccess.psutil.process_iter', return_value=[unrelated]), \
             patch('assistant.tools.WebAccess.psutil.wait_procs') as wait_mock:
            wa._kill_orphan_chrome_using_profile()

        unrelated.terminate.assert_not_called()
        wait_mock.assert_not_called()

    def test_handles_disappeared_process_during_iteration(self):
        import psutil as _psutil
        wa = self._make_wa("/tmp/p")
        gone = Mock()
        type(gone).info = property(lambda _self: (_ for _ in ()).throw(_psutil.NoSuchProcess(999)))
        survivor = self._proc(555, ['/usr/bin/chrome', '--user-data-dir=/tmp/p'])

        with patch('assistant.tools.WebAccess.psutil.process_iter', return_value=[gone, survivor]), \
             patch('assistant.tools.WebAccess.psutil.wait_procs', return_value=([survivor], [])):
            wa._kill_orphan_chrome_using_profile()  # should not raise

        survivor.terminate.assert_called_once()

    def test_oserror_from_dying_process_skips_it(self):
        # macOS raises OSError [Errno 0] (sysctl KERN_PROCARGS2) when reading the
        # cmdline of a process that is mid-death — e.g. one browse_close just
        # SIGKILLed. The sweep must skip it, not blow up the browser init.
        wa = self._make_wa("/tmp/p")
        dying = Mock()
        type(dying).info = property(
            lambda _self: (_ for _ in ()).throw(OSError(0, "Undefined error: 0")))
        survivor = self._proc(666, ['/usr/bin/chrome', '--user-data-dir=/tmp/p'])

        with patch('assistant.tools.WebAccess.psutil.process_iter', return_value=[dying, survivor]), \
             patch('assistant.tools.WebAccess.psutil.wait_procs', return_value=([survivor], [])):
            wa._kill_orphan_chrome_using_profile()  # should not raise

        survivor.terminate.assert_called_once()

    def test_oserror_from_process_iter_itself_is_contained(self):
        wa = self._make_wa("/tmp/p")

        def exploding_iter(_attrs):
            raise OSError(0, "Undefined error: 0")
            yield  # pragma: no cover

        with patch('assistant.tools.WebAccess.psutil.process_iter', side_effect=exploding_iter):
            wa._kill_orphan_chrome_using_profile()  # should not raise


@pytest.mark.unit
class TestExtractIframeTexts:
    def test_enumeration_does_not_use_polling_select_all(self):
        # select_all polls up to 10s when a page has no iframes, which turned
        # every browse_extract on an iframe-less page into a fixed 10s stall.
        wa = WebAccess.__new__(WebAccess)
        tab = Mock()
        tab.query_selector_all = AsyncMock(return_value=[])
        tab.select_all = AsyncMock()

        result = asyncio.run(wa._async_extract_iframe_texts(tab, 'https://x.se'))

        assert result == ""
        tab.query_selector_all.assert_awaited_once_with('iframe')
        tab.select_all.assert_not_called()


@pytest.mark.unit
class TestPdfDetection:
    """Tests for PDF URL detection."""

    def test_pdf_extension_detected(self):
        assert WebAccess._is_pdf_url("https://example.com/report.pdf") is True

    def test_pdf_extension_with_query_params(self):
        assert WebAccess._is_pdf_url("https://example.com/report.pdf?token=abc") is True

    def test_pdf_extension_case_insensitive(self):
        assert WebAccess._is_pdf_url("https://example.com/Report.PDF") is True

    def test_pdf_in_path_but_not_extension(self):
        assert WebAccess._is_pdf_url("https://example.com/pdf/viewer") is False

@pytest.mark.unit
class TestPdfFetching:
    """Tests for PDF content extraction."""

    @staticmethod
    def _make_wa():
        with patch.object(WebAccess, '__init__', lambda self: None):
            wa = WebAccess()
            return wa

    @patch('assistant.tools.WebAccess.requests.get')
    def test_pdf_too_large_via_content_length(self, mock_get):
        wa = self._make_wa()

        mock_response = Mock()
        mock_response.headers = {'Content-Length': str(60 * 1024 * 1024)}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = wa._fetch_pdf_content("https://example.com/huge.pdf")
        assert "too large" in result

    @patch('assistant.tools.WebAccess.requests.get')
    def test_pdf_too_large_via_streaming(self, mock_get):
        wa = self._make_wa()

        mock_response = Mock()
        mock_response.headers = {}
        mock_response.raise_for_status = Mock()
        chunk = b'x' * 1024
        mock_response.iter_content.return_value = [chunk] * 5
        mock_get.return_value = mock_response

        # No Content-Length to refuse on, so the cap is met while streaming; a cap of 3 kB
        # trips it without building the real 50 MB in memory.
        with patch('assistant.tools.PdfExtractor.MAX_PDF_SIZE', 3 * 1024):
            result = wa._fetch_pdf_content("https://example.com/huge.pdf")
        assert "too large" in result

    @patch('assistant.tools.WebAccess.requests.get')
    def test_pdf_network_error(self, mock_get):
        wa = self._make_wa()
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        result = wa._fetch_pdf_content("https://example.com/doc.pdf")
        assert "ERROR" in result

    @patch.object(WebAccess, '_fetch_pdf_content', return_value="PDF text content")
    @patch.object(WebAccess, '__init__', lambda self: None)
    def test_fetch_webpage_routes_pdf_to_handler(self, mock_pdf):
        wa = WebAccess()
        result = wa.fetch_webpage("https://example.com/report.pdf")
        assert result == "PDF text content"
        mock_pdf.assert_called_once_with("https://example.com/report.pdf")


@pytest.mark.unit
class TestFormatSnapshotLine:
    def _make_wa(self):
        wa = WebAccess.__new__(WebAccess)
        return wa

    def test_text_input(self):
        wa = self._make_wa()
        rec = {'tag': 'input', 'type': 'text', 'label': 'Username', 'value': 'alice'}
        result = wa._format_snapshot_line(0, rec)
        assert result == "[0] input[text] \"Username\" value='alice'"

    def test_empty_text_input(self):
        wa = self._make_wa()
        rec = {'tag': 'input', 'type': 'text', 'label': 'Search', 'value': ''}
        result = wa._format_snapshot_line(3, rec)
        assert result == '[3] input[text] "Search"'

    def test_checkbox_input(self):
        wa = self._make_wa()
        rec = {'tag': 'input', 'type': 'checkbox', 'label': 'Accept terms', 'checked': True}
        result = wa._format_snapshot_line(1, rec)
        assert result == '[1] input[checkbox] "Accept terms" checked=True'

    def test_radio_input_unchecked(self):
        wa = self._make_wa()
        rec = {'tag': 'input', 'type': 'radio', 'label': 'Option A', 'checked': False}
        result = wa._format_snapshot_line(2, rec)
        assert result == '[2] input[radio] "Option A" checked=False'

    def test_textarea(self):
        wa = self._make_wa()
        rec = {'tag': 'textarea', 'type': '', 'label': 'Comment', 'value': 'hello'}
        result = wa._format_snapshot_line(0, rec)
        assert result == "[0] textarea \"Comment\" value='hello'"

    def test_select(self):
        wa = self._make_wa()
        rec = {'tag': 'select', 'label': 'Country', 'value': 'SE',
               'options': [{'text': 'Sweden', 'value': 'SE'}, {'text': 'Norway', 'value': 'NO'}]}
        result = wa._format_snapshot_line(4, rec)
        assert '[4] select "Country"' in result
        assert "value='SE'" in result
        assert 'Sweden,Norway' in result

    def test_select_many_options_not_truncated(self):
        wa = self._make_wa()
        opts = [{'text': f'Option {i}', 'value': str(i)} for i in range(50)]
        rec = {'tag': 'select', 'label': 'Big list', 'value': '0', 'options': opts}
        result = wa._format_snapshot_line(0, rec)
        assert ',...' not in result
        assert 'Option 0' in result
        assert 'Option 49' in result

    def test_button(self):
        wa = self._make_wa()
        rec = {'tag': 'button', 'label': 'Submit'}
        result = wa._format_snapshot_line(5, rec)
        assert result == '[5] button "Submit"'

    def test_role_button(self):
        wa = self._make_wa()
        rec = {'tag': 'div', 'role': 'button', 'label': 'Click me'}
        result = wa._format_snapshot_line(0, rec)
        assert result == '[0] button "Click me"'

    def test_link(self):
        wa = self._make_wa()
        rec = {'tag': 'a', 'label': 'Home page', 'href': 'https://example.com'}
        result = wa._format_snapshot_line(6, rec)
        assert result == '[6] link "Home page" -> https://example.com'

    def test_role_link(self):
        wa = self._make_wa()
        rec = {'tag': 'span', 'role': 'link', 'label': 'More info', 'href': '/info'}
        result = wa._format_snapshot_line(0, rec)
        assert result == '[0] link "More info" -> /info'

    def test_contenteditable(self):
        wa = self._make_wa()
        rec = {'tag': 'div', 'contenteditable': True, 'label': 'Rich editor', 'value': 'draft'}
        result = wa._format_snapshot_line(7, rec)
        assert result == "[7] editable \"Rich editor\" value='draft'"

    def test_role_textbox(self):
        wa = self._make_wa()
        rec = {'tag': 'div', 'role': 'textbox', 'label': 'Notes', 'value': ''}
        result = wa._format_snapshot_line(0, rec)
        assert result == "[0] editable \"Notes\" value=''"

    def test_generic_element(self):
        wa = self._make_wa()
        rec = {'tag': 'summary', 'label': 'Details'}
        result = wa._format_snapshot_line(8, rec)
        assert result == '[8] summary "Details"'

    def test_no_label_fallback(self):
        wa = self._make_wa()
        rec = {'tag': 'button', 'label': ''}
        result = wa._format_snapshot_line(0, rec)
        assert result == '[0] button "(no label)"'


@pytest.mark.unit
class TestBrowseActValidation:
    """browse_act input validation. The async path is exercised via the loop_thread mock."""

    @pytest.fixture(autouse=True)
    def _healthy_browser(self):
        # The health gate probes/reinits Chrome on every command; keep it a no-op here.
        with patch.object(WebAccess, '_async_ensure_browser', new=AsyncMock(return_value=True)):
            yield

    def _make_wa(self, with_elements=True):
        wa = WebAccess.__new__(WebAccess)
        wa._browser = Mock()
        wa._tab = Mock()
        wa._browse_session_url = "https://example.com"
        wa._browse_snapshot_frame_ids = [None] if with_elements else []
        wa._browse_pdf_text = None
        # Run async validation directly via asyncio.run since the validation
        # paths return immediately without awaiting anything.
        import asyncio
        wa._browse_lock = asyncio.Lock()
        wa._loop_thread = Mock()
        wa._loop_thread.submit = lambda coro, timeout=None: asyncio.run(coro)
        return wa

    def test_no_browser(self):
        wa = self._make_wa()
        wa._tab = None
        with patch.object(WebAccess, '_async_ensure_browser', new=AsyncMock(return_value=False)):
            result = wa.browse_act("0")
        assert "ERROR" in result
        assert "not available" in result

    def test_no_snapshot(self):
        wa = self._make_wa(with_elements=False)
        result = wa.browse_act("0")
        assert "ERROR" in result
        assert "no current snapshot" in result

    def test_non_integer_index(self):
        wa = self._make_wa()
        result = wa.browse_act("abc")
        assert "ERROR" in result
        assert "not a valid index" in result


@pytest.mark.unit
class TestBrowseEval:
    """browse_eval JSON serialization, truncation, and error paths."""

    @pytest.fixture(autouse=True)
    def _healthy_browser(self):
        with patch.object(WebAccess, '_async_ensure_browser', new=AsyncMock(return_value=True)):
            yield

    def _make_wa(self, with_session=True):
        wa = WebAccess.__new__(WebAccess)
        wa._browser = Mock()
        wa._tab = Mock()
        wa._browse_session_url = "https://example.com" if with_session else None
        wa._browse_pdf_text = None
        wa._dialog_notes = []
        import asyncio
        wa._browse_lock = asyncio.Lock()
        wa._loop_thread = Mock()
        wa._loop_thread.submit = lambda coro, timeout=None: asyncio.run(coro)
        return wa

    def test_no_browser(self):
        wa = self._make_wa()
        wa._tab = None
        with patch.object(WebAccess, '_async_ensure_browser', new=AsyncMock(return_value=False)):
            assert "ERROR" in wa.browse_eval("return 1")

    def test_no_session(self):
        wa = self._make_wa(with_session=False)
        result = wa.browse_eval("return 1")
        assert "ERROR" in result and "no browse session" in result

    def test_empty_js(self):
        wa = self._make_wa()
        result = wa.browse_eval("   ")
        assert "ERROR" in result and "needs a JavaScript" in result

    def test_string_returned_as_is(self):
        wa = self._make_wa()
        async def fake_eval(_tab, _expr):
            return "hello"
        with patch.object(WebAccess, '_evaluate', side_effect=fake_eval):
            assert wa.browse_eval("return 'hello'") == "hello"

    def test_dict_returned_as_json(self):
        wa = self._make_wa()
        async def fake_eval(_tab, _expr):
            return {"token": "abc", "n": 3}
        with patch.object(WebAccess, '_evaluate', side_effect=fake_eval):
            import json
            out = wa.browse_eval("return {}")
            assert json.loads(out) == {"token": "abc", "n": 3}

    def test_none_returned_as_null_string(self):
        wa = self._make_wa()
        async def fake_eval(_tab, _expr):
            return None
        with patch.object(WebAccess, '_evaluate', side_effect=fake_eval):
            assert wa.browse_eval("return null") == "null"

    def test_truncation_past_limit(self):
        wa = self._make_wa()
        big = "x" * (WebAccess._BROWSE_EVAL_MAX_CHARS + 100)
        async def fake_eval(_tab, _expr):
            return big
        with patch.object(WebAccess, '_evaluate', side_effect=fake_eval):
            out = wa.browse_eval("return ''")
            assert out.endswith("(truncated)")
            assert len(out) < len(big)

    def test_in_page_throw_returned_as_error(self):
        wa = self._make_wa()
        async def fake_eval(_tab, _expr):
            return {"__floxy_eval_error": "TypeError: oops"}
        with patch.object(WebAccess, '_evaluate', side_effect=fake_eval):
            out = wa.browse_eval("return foo.bar")
            assert "ERROR" in out and "TypeError" in out

    def test_evaluate_raises(self):
        wa = self._make_wa()
        async def fake_eval(_tab, _expr):
            raise RuntimeError("boom")
        with patch.object(WebAccess, '_evaluate', side_effect=fake_eval):
            out = wa.browse_eval("return 1")
            assert "ERROR" in out and "boom" in out

    def test_wrapper_tries_expression_form_before_block(self):
        """Regression: bare expressions like `window.location.href` must work, not just blocks with return."""
        wa = self._make_wa()
        captured = {}
        async def fake_eval(_tab, expr):
            captured['expr'] = expr
            return 1
        with patch.object(WebAccess, '_evaluate', side_effect=fake_eval):
            wa.browse_eval("window.location.href")
            expr = captured['expr']
            # Expression form must appear before the block-form fallback.
            assert "'return (' + __body + ');'" in expr
            assert expr.index("'return (' + __body + ');'") < expr.index(
                "new AsyncFn(__body)"
            )

    def test_body_is_json_encoded_in_wrapper(self):
        """Regression: body must be JSON-encoded so quotes/backticks/newlines can't break out of the wrapper."""
        wa = self._make_wa()
        captured = {}
        async def fake_eval(_tab, expr):
            captured['expr'] = expr
            return 1
        # A body containing a double quote and a newline would close out a naively-inlined string.
        nasty = 'return "he said \\"hi\\""'
        with patch.object(WebAccess, '_evaluate', side_effect=fake_eval):
            wa.browse_eval(nasty)
        import json
        # The body must appear exactly as a JSON-encoded string literal in the wrapper.
        assert json.dumps(nasty) in captured['expr']

    def test_wrapper_invokes_body_exactly_once(self):
        """Regression for c91c6d41: a non-idempotent body (e.g. fetch POST) must
        run exactly once. The old wrapper awaited the expression-form invocation,
        then on *any* throw (including runtime errors from the body) re-awaited
        a block-form invocation — duplicating POSTs and side effects. The fix
        decides the form at construction time, then awaits the chosen function
        once. The expression-form-vs-block decision and the body invocation
        must be in separate try/catch blocks, and no AsyncFn construction may
        sit in the catch of the invocation."""
        wa = self._make_wa()
        captured = {}
        async def fake_eval(_tab, expr):
            captured['expr'] = expr
            return 1
        with patch.object(WebAccess, '_evaluate', side_effect=fake_eval):
            wa.browse_eval("return fetch('/x', {method:'POST'})")
        expr = captured['expr']
        # The buggy form invoked the constructor inline: `await (new AsyncFn(...))()`.
        # Neither that nor any variant may appear — construction is unawaited.
        assert "await (new AsyncFn" not in expr
        assert "await new AsyncFn" not in expr
        # And the body must be awaited via the shared handle exactly once.
        assert expr.count("await __fn()") == 1
        # Nothing after the invocation may re-construct an AsyncFn (i.e. retry
        # the body on a runtime throw). The runtime catch must only tag the error.
        tail = expr[expr.index("return await __fn()"):]
        assert "new AsyncFn" not in tail


@pytest.mark.unit
class TestNormalizeUrl:
    def test_adds_https_prefix(self):
        assert WebAccess._normalize_url("example.com") == "https://example.com"

    def test_protocol_relative(self):
        assert WebAccess._normalize_url("//example.com/path") == "https://example.com/path"

    def test_keeps_http(self):
        assert WebAccess._normalize_url("http://example.com") == "http://example.com"

    def test_empty_string(self):
        assert WebAccess._normalize_url("") == ""

    def test_whitespace_stripped(self):
        assert WebAccess._normalize_url("  example.com  ") == "https://example.com"

    def test_none_returns_empty(self):
        assert WebAccess._normalize_url(None) == ""


@pytest.mark.unit
class TestBrowseClose:
    def test_browse_close_resets_state(self):
        wa = WebAccess.__new__(WebAccess)
        wa._browser = None
        wa._tab = Mock()
        wa._browse_session_url = "https://example.com"
        wa._browse_snapshot_frame_ids = [None]
        wa._popup_parents = []
        wa._known_page_targets = set()
        wa._pending_target_notes = []
        wa._dialog_notes = []
        wa._browse_lock = asyncio.Lock()
        wa._loop_thread = Mock()
        wa._loop_thread.submit = lambda coro, timeout=None: asyncio.run(coro)
        result = wa.browse_close()
        assert "closed" in result.lower()
        assert wa._browse_session_url is None
        assert wa._browse_snapshot_frame_ids == []


@pytest.mark.unit
class TestJsEscape:
    def test_escapes_single_quote(self):
        assert WebAccess._js_escape("it's") == "it\\'s"

    def test_escapes_double_quote(self):
        assert WebAccess._js_escape('he said "hi"') == 'he said \\"hi\\"'

    def test_escapes_backslash(self):
        assert WebAccess._js_escape("a\\b") == "a\\\\b"

    def test_escapes_newline(self):
        assert WebAccess._js_escape("line1\nline2") == "line1\\nline2"


@pytest.mark.unit
class TestLoopThread:
    """The private asyncio loop thread that bridges sync callers to nodriver."""

    def test_start_and_stop(self):
        from assistant.tools.WebAccess import _LoopThread
        lt = _LoopThread()
        lt.start()
        assert lt.is_running()
        lt.stop()
        assert not lt.is_running()

    def test_submit_returns_coro_result(self):
        from assistant.tools.WebAccess import _LoopThread
        lt = _LoopThread()
        lt.start()
        try:
            async def add(a, b):
                return a + b
            assert lt.submit(add(2, 3), timeout=5) == 5
        finally:
            lt.stop()

    def test_submit_propagates_exception(self):
        from assistant.tools.WebAccess import _LoopThread
        lt = _LoopThread()
        lt.start()
        try:
            async def boom():
                raise ValueError("nope")
            with pytest.raises(ValueError, match="nope"):
                lt.submit(boom(), timeout=5)
        finally:
            lt.stop()


@pytest.mark.unit
class TestBrowseScreenshot:
    """Tests for browse_screenshot file save logic (mocks the browser)."""

    @pytest.fixture(autouse=True)
    def _healthy_browser(self):
        with patch.object(WebAccess, '_async_ensure_browser', new=AsyncMock(return_value=True)):
            yield

    def _bare_webaccess(self):
        wa = WebAccess.__new__(WebAccess)
        wa._browser = None
        wa._tab = None
        wa._browse_session_url = None
        wa._browse_pdf_text = None
        wa._last_screenshot_bytes = None
        wa._last_screenshot_path = None
        wa._popup_parents = []
        wa._known_page_targets = set()
        wa._pending_target_notes = []
        wa._dialog_notes = []
        wa._browse_lock = asyncio.Lock()
        wa._loop_thread = Mock()
        wa._loop_thread.submit = lambda coro, timeout=None: asyncio.run(coro)
        return wa

    def _real_png_bytes(self):
        from PIL import Image
        from io import BytesIO
        img = Image.new("RGB", (32, 24), color=(120, 200, 80))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_saves_png_and_returns_blocks(self, tmp_path):
        png = self._real_png_bytes()
        wa = self._bare_webaccess()
        wa._tab = Mock()
        wa._tab.target = Mock(url="https://example.com/foo")
        wa._tab.send = AsyncMock(return_value=base64.b64encode(png).decode('ascii'))
        wa._browse_session_url = "https://example.com/foo"

        with patch.object(WebAccess, '_screenshot_dir', staticmethod(lambda: str(tmp_path))):
            result = asyncio.run(wa._async_browse_screenshot())

        assert isinstance(result, list)
        assert len(result) == 2
        text_block, image_block = result
        assert text_block["type"] == "text"
        assert text_block["text"].startswith("Saved screenshot of https://example.com/foo to ")
        assert str(tmp_path) in text_block["text"]
        assert image_block["type"] == "image"
        assert image_block["source"]["type"] == "base64"
        assert image_block["source"]["media_type"] in ("image/jpeg", "image/png")
        assert len(image_block["source"]["data"]) > 0

        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].suffix == ".png"
        assert files[0].read_bytes() == png

        cached_bytes, cached_path = wa.get_last_screenshot()
        assert cached_bytes == png
        assert cached_path == str(files[0])

    def test_browse_close_clears_cached_screenshot(self):
        wa = self._bare_webaccess()
        wa._last_screenshot_bytes = b"some_png"
        wa._last_screenshot_path = "/tmp/x.png"
        wa.browse_close()
        assert wa.get_last_screenshot() == (None, None)

    def test_no_browser_returns_error(self):
        wa = self._bare_webaccess()
        wa._browse_session_url = "https://example.com"
        with patch.object(WebAccess, '_async_ensure_browser', new=AsyncMock(return_value=False)):
            result = asyncio.run(wa._async_browse_screenshot())
        assert result.startswith("ERROR: browser is not available")

    def test_no_session_returns_error(self):
        wa = self._bare_webaccess()
        wa._tab = Mock()
        result = asyncio.run(wa._async_browse_screenshot())
        assert result.startswith("ERROR: no browse session")

    def test_filename_uses_url_slug(self):
        name = WebAccess._screenshot_filename("https://example.com/path/to/page?x=1")
        assert name.endswith(".png")
        assert "example.com" in name
        assert "/" not in name and "?" not in name and "=" not in name

@pytest.mark.unit
class TestPdfSessionGuards:
    """PDF sessions block browse_act/inspect/eval/screenshot and return cached text on extract."""

    def _make_pdf_wa(self):
        wa = WebAccess.__new__(WebAccess)
        wa._browser = None
        wa._tab = Mock()
        wa._browse_session_url = "https://example.com/doc.pdf"
        wa._browse_pdf_text = "Extracted PDF content for testing purposes."
        wa._browse_snapshot_frame_ids = []
        wa._browse_snapshot_labels = []
        wa._last_screenshot_bytes = None
        wa._last_screenshot_path = None
        wa._popup_parents = []
        wa._known_page_targets = set()
        wa._pending_target_notes = []
        wa._dialog_notes = []
        import asyncio
        wa._browse_lock = asyncio.Lock()
        wa._loop_thread = Mock()
        wa._loop_thread.submit = lambda coro, timeout=None: asyncio.run(coro)
        return wa

    def test_pdf_session_error_returns_none_without_pdf(self):
        wa = WebAccess.__new__(WebAccess)
        wa._browse_pdf_text = None
        assert wa._pdf_session_error("test") is None

    def test_browse_act_blocked_on_pdf(self):
        wa = self._make_pdf_wa()
        result = wa.browse_act("0")
        assert "ERROR" in result
        assert "PDF" in result

    def test_browse_eval_blocked_on_pdf(self):
        wa = self._make_pdf_wa()
        result = wa.browse_eval("return 1")
        assert "ERROR" in result
        assert "PDF" in result

    def test_browse_screenshot_blocked_on_pdf(self):
        wa = self._make_pdf_wa()
        result = asyncio.run(wa._async_browse_screenshot())
        assert "ERROR" in result
        assert "PDF" in result

    def test_browse_inspect_blocked_on_pdf(self):
        wa = self._make_pdf_wa()
        result = asyncio.run(wa._async_browse_inspect("0"))
        assert "ERROR" in result
        assert "PDF" in result

    def test_browse_extract_returns_cached_pdf_text(self):
        wa = self._make_pdf_wa()
        result = asyncio.run(wa._async_browse_extract())
        assert result == "Extracted PDF content for testing purposes."

    def test_browse_close_clears_pdf_state(self):
        wa = self._make_pdf_wa()
        wa.browse_close()
        assert wa._browse_pdf_text is None
        assert wa._browse_session_url is None

    def test_reset_browse_session_state_clears_snapshot(self):
        wa = WebAccess.__new__(WebAccess)
        wa._popup_parents = []
        wa._known_page_targets = set()
        wa._pending_target_notes = []
        wa._dialog_notes = []
        wa._browse_snapshot_frame_ids = [None]
        wa._browse_snapshot_labels = ["link"]
        wa._last_screenshot_bytes = b"png"
        wa._last_screenshot_path = "/tmp/x.png"
        wa._reset_browse_session_state()
        assert wa._browse_snapshot_frame_ids == []
        assert wa._browse_snapshot_labels == []
        assert wa._last_screenshot_bytes is None
        assert wa._last_screenshot_path is None


@pytest.mark.unit
class TestMaybeRetryInit:
    """WebAccess._async_maybe_retry_init: rate-limited browser self-heal.

    Covers the recovery path added so a dead Chrome session re-inits on the
    next fetch instead of staying down until process restart. The contract:
    attempt re-init only when the browser is actually down, no other attempt
    is in flight, and the backoff window has elapsed — and always leave
    _init_in_progress reset so the next caller isn't locked out.
    """

    def _make_wa(self):
        wa = WebAccess.__new__(WebAccess)
        wa._browser = None
        wa._init_in_progress = False
        wa._last_init_failure_ts = 0.0
        return wa

    def test_noop_when_browser_already_up(self):
        wa = self._make_wa()
        wa._browser = Mock()
        with patch.object(WebAccess, '_async_initialize_driver', new_callable=AsyncMock) as init:
            asyncio.run(wa._async_maybe_retry_init())
        init.assert_not_called()

    def test_noop_when_init_already_in_progress(self):
        wa = self._make_wa()
        wa._init_in_progress = True
        with patch.object(WebAccess, '_async_initialize_driver', new_callable=AsyncMock) as init:
            asyncio.run(wa._async_maybe_retry_init())
        init.assert_not_called()

    def test_noop_within_backoff_window(self):
        wa = self._make_wa()
        wa._last_init_failure_ts = time.time()  # failed just now
        with patch.object(WebAccess, '_async_initialize_driver', new_callable=AsyncMock) as init:
            asyncio.run(wa._async_maybe_retry_init())
        init.assert_not_called()

    def test_retries_once_backoff_elapsed(self):
        wa = self._make_wa()
        wa._last_init_failure_ts = time.time() - WebAccess._INIT_RETRY_BACKOFF - 1
        with patch.object(WebAccess, '_async_initialize_driver', new_callable=AsyncMock) as init:
            asyncio.run(wa._async_maybe_retry_init())
        init.assert_awaited_once()

    def test_clears_in_progress_flag_after_successful_recovery(self):
        wa = self._make_wa()
        wa._last_init_failure_ts = time.time() - WebAccess._INIT_RETRY_BACKOFF - 1

        async def fake_init(self):
            self._browser = Mock()

        with patch.object(WebAccess, '_async_initialize_driver', fake_init):
            asyncio.run(wa._async_maybe_retry_init())
        assert wa._browser is not None
        assert wa._init_in_progress is False

    def test_clears_in_progress_flag_after_failed_recovery(self):
        wa = self._make_wa()
        wa._last_init_failure_ts = time.time() - WebAccess._INIT_RETRY_BACKOFF - 1

        async def fake_init(self):
            self._browser = None  # init failed again

        with patch.object(WebAccess, '_async_initialize_driver', fake_init):
            asyncio.run(wa._async_maybe_retry_init())
        assert wa._browser is None
        assert wa._init_in_progress is False

    def test_init_exception_is_contained_and_recorded_as_failure(self):
        # A raised init (e.g. the macOS sysctl race when Chrome was just killed)
        # must not escape to the calling browse command, and must arm the
        # backoff so the next command doesn't hammer init again.
        wa = self._make_wa()
        wa._tab = Mock()

        async def raising_init(self):
            raise OSError(0, "Undefined error: 0")

        with patch.object(WebAccess, '_async_initialize_driver', raising_init):
            asyncio.run(wa._async_maybe_retry_init())  # must not raise
        assert wa._browser is None
        assert wa._tab is None
        assert wa._init_in_progress is False
        assert time.time() - wa._last_init_failure_ts < 5


class TestBoilerplateGuard:
    """The boilerplate class rule must not eat a layout wrapper that holds the page."""

    REDDIT_SHAPED = """
    <html><body>
        <shreddit-app>
            <data-protection-consent hidden>
                <template>
                    <div class="consent-dialog">
                        Reddit and its partners use cookies. Accept all Reject Optional Cookies
                    </div>
                </template>
            </data-protection-consent>
            <div class="main-container grid-container fixed-sidebar">
                <main>
                    <h1>Why does my sourdough collapse?</h1>
                    <p>Post body: it rises fine and then falls in the oven.</p>
                    <div class="comment">Over-proofed. Shorten the bulk ferment.</div>
                    <div class="comment">Score deeper and bake hotter.</div>
                </main>
                <div class="right-sidebar"><a href="/r/Sourdough">About r/Sourdough</a></div>
            </div>
        </shreddit-app>
    </body></html>
    """

    def test_layout_wrapper_with_sidebar_class_is_kept(self):
        result = WebAccess.extract_text_with_links(self.REDDIT_SHAPED, "https://www.reddit.com/r/Sourdough/comments/x/y/")
        assert "Why does my sourdough collapse?" in result
        assert "Shorten the bulk ferment" in result
        assert "bake hotter" in result

    def test_template_markup_never_leaks(self):
        result = WebAccess.extract_text_with_links(self.REDDIT_SHAPED, "https://www.reddit.com/r/Sourdough/comments/x/y/")
        assert "Reject Optional Cookies" not in result
        assert "partners use cookies" not in result

    def test_small_sidebar_is_still_removed(self):
        result = WebAccess.extract_text_with_links(self.REDDIT_SHAPED, "https://www.reddit.com/r/Sourdough/comments/x/y/")
        assert "About r/Sourdough" not in result

    def test_genuine_modal_is_still_removed(self):
        html = """
        <html><body>
            <div class="modal">Subscribe to our newsletter!</div>
            <main><p>The article text, which is long enough to be the page's bulk of content.</p></main>
        </body></html>
        """
        result = WebAccess.extract_text_with_links(html, "https://example.com")
        assert "Subscribe to our newsletter" not in result
        assert "article text" in result
