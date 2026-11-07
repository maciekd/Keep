"""The extractor's links: tracking parameters stripped, a repeated link rendered once."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from assistant.tools.WebAccess import WebAccess


def _page(body: str) -> str:
    return f"<html><body><main><div>{body}</div></main></body></html>"


class TestCleanLink:
    def test_generic_tracking_parameters_go_and_the_rest_keeps_its_order(self):
        url = "https://example.com/p?b=2&utm_source=mail&a=1&fbclid=xyz&gclid=q"
        assert WebAccess.clean_link(url) == "https://example.com/p?b=2&a=1"

    def test_a_link_without_tracking_is_returned_unchanged(self):
        for url in ("https://example.com/p?foo=bar&baz=qux", "https://example.com/p#frag",
                    "https://example.com/p?q=a%20b"):
            assert WebAccess.clean_link(url) == url

    def test_site_parameters_apply_to_that_host_and_its_subdomains_only(self):
        query = "aid=304142&label=mkt-abc&checkin=2027-04-07&srpvid=ff&group_adults=8"
        assert (WebAccess.clean_link(f"https://www.booking.com/hotel/x.html?{query}")
                == "https://www.booking.com/hotel/x.html?checkin=2027-04-07&group_adults=8")
        assert (WebAccess.clean_link(f"https://booking.com/hotel/x.html?{query}")
                == "https://booking.com/hotel/x.html?checkin=2027-04-07&group_adults=8")
        # `aid` and `label` mean something on another site
        other = f"https://notbooking.com/article?{query}"
        assert WebAccess.clean_link(other) == other
        lookalike = f"https://mybooking.com/x?{query}"
        assert WebAccess.clean_link(lookalike) == lookalike

    def test_a_blank_value_survives(self):
        assert WebAccess.clean_link("https://example.com/p?flag=&utm_x=1") == "https://example.com/p?flag="

    def test_the_fragment_outlives_the_stripping(self):
        """The query is rebuilt when something goes; the anchor after it is not part of the query."""
        assert WebAccess.clean_link("https://example.com/p?x=1&utm_source=mail#rooms") == "https://example.com/p?x=1#rooms"


class TestExtractedLinks:
    def test_links_lose_their_tracking_parameters(self):
        text = WebAccess.extract_text_with_links(
            _page('<a href="/h?aid=1&id=5">Hotel</a>'), "https://www.booking.com/")
        assert '[Hotel](/h?id=5)' in text
        assert "aid=" not in text

    def test_a_repeated_link_is_rendered_once_and_its_later_texts_stay(self):
        text = WebAccess.extract_text_with_links(
            _page('<a href="/h?aid=1&id=5">Hotel</a> <a href="/h?aid=2&id=5">Hotel</a> '
                  '<a href="/h?id=5">SEK 35,622</a> <a href="/other">Other</a>'),
            "https://www.booking.com/")
        assert text.count('](/') == 2
        assert text.count('](/h?id=5)') == 1
        assert "SEK 35,622" in text
        assert '[Other](/other)' in text

    def test_distinct_links_are_all_kept(self):
        text = WebAccess.extract_text_with_links(
            _page('<a href="/a">A</a> <a href="/b">B</a> <a href="/a#x">A again</a>'),
            "https://example.com/")
        assert text.count('](/') == 3

    def test_the_seen_set_is_per_page(self):
        page = _page('<a href="/a">A</a>')
        first = WebAccess.extract_text_with_links(page, "https://example.com/")
        second = WebAccess.extract_text_with_links(page, "https://example.com/")
        assert first == second and '[A](/a)' in second


class TestLinkRendering:
    """A link is `[text](url)`: a path when it stays on the page's site, absolute when it
    leaves it; a link that leads nowhere the LLM would go is its text alone."""

    def test_a_link_on_the_pages_own_site_is_a_root_relative_path(self):
        text = WebAccess.extract_text_with_links(
            _page('<a href="/wiki/Raptor?x=1#engine">Raptor</a>'), "https://en.wikipedia.org/wiki/Starship")
        assert "[Raptor](/wiki/Raptor?x=1#engine)" in text

    def test_a_link_to_another_site_keeps_the_whole_url(self):
        text = WebAccess.extract_text_with_links(
            _page('<a href="https://www.spacex.com/updates">updates</a>'), "https://en.wikipedia.org/wiki/Starship")
        assert "[updates](https://www.spacex.com/updates)" in text

    def test_a_link_to_the_site_root_is_a_slash(self):
        assert WebAccess.render_link("https://example.com", "Home", "https://example.com") == "[Home](/)"

    def test_a_lookalike_host_is_not_the_same_site(self):
        assert WebAccess.render_link("https://example.com.evil.test/x", "X", "https://example.com") \
            == "[X](https://example.com.evil.test/x)"

    def test_a_same_page_anchor_is_its_text_alone(self):
        text = WebAccess.extract_text_with_links(
            _page('fact <a href="#cite_note-3">[3]</a> and <a href="/wiki/Raptor#History">history</a>'),
            "https://en.wikipedia.org/wiki/Starship")
        assert "fact [3] and" in text
        assert "[history](/wiki/Raptor#History)" in text

    def test_wiki_housekeeping_links_are_their_text_alone(self):
        text = WebAccess.extract_text_with_links(
            _page('<a href="/w/index.php?title=Starship&action=edit&section=2">edit</a> '
                  '<a href="/wiki/Template:Infobox">T</a> <a href="/wiki/File:Starship.jpg">F</a> '
                  '<a href="/wiki/Raptor">Raptor</a>'),
            "https://en.wikipedia.org/wiki/Starship")
        assert "](" not in text.replace("[Raptor](/wiki/Raptor)", "")
        assert "[Raptor](/wiki/Raptor)" in text


class TestPathResolution:
    """A path the LLM hands back is resolved against the page it was read from."""

    def _wa(self):
        return WebAccess.__new__(WebAccess)

    def test_a_path_is_resolved_against_the_last_fetched_page(self):
        wa = self._wa()
        wa._link_base_url = "https://en.wikipedia.org/wiki/Starship"
        assert wa._resolve_page_link("/wiki/Raptor") == "https://en.wikipedia.org/wiki/Raptor"

    def test_a_full_url_and_a_protocol_relative_one_pass_through(self):
        wa = self._wa()
        wa._link_base_url = "https://en.wikipedia.org/wiki/Starship"
        assert wa._resolve_page_link("https://example.com/a") == "https://example.com/a"
        assert wa._resolve_page_link("//cdn.example.com/a") == "//cdn.example.com/a"

    def test_a_path_with_no_page_read_yet_passes_through(self):
        assert self._wa()._resolve_page_link("/wiki/Raptor") == "/wiki/Raptor"

    def test_fetch_webpage_keeps_the_resolved_url_as_the_page_key(self):
        wa = self._wa()
        wa._link_base_url = "https://en.wikipedia.org/wiki/Starship"
        with patch.object(wa, "_fetch_webpage_text", return_value="hello") as fetch:
            wa.fetch_webpage("/wiki/Raptor")
        fetch.assert_called_once_with("https://en.wikipedia.org/wiki/Raptor")


class TestBrowseLinkBase:
    """A path the LLM reads off a browse snapshot resolves against the page the browser
    shows, not the page the last open_url read."""

    def _wa(self, tab_url):
        wa = WebAccess.__new__(WebAccess)
        wa._link_base_url = "https://news.example/front"
        wa._tab = SimpleNamespace(target=SimpleNamespace(url=tab_url))
        return wa

    def test_the_tab_page_becomes_the_base(self):
        wa = self._wa("https://shop.example/catalog?page=1")
        wa._anchor_links_to_tab("https://shop.example/")
        assert wa._resolve_page_link("/page/2") == "https://shop.example/page/2"

    def test_the_requested_url_is_the_base_when_the_tab_has_none(self):
        wa = self._wa("")
        wa._anchor_links_to_tab("https://shop.example/")
        assert wa._resolve_page_link("/page/2") == "https://shop.example/page/2"

    def test_a_chrome_error_page_never_becomes_the_base(self):
        # a navigation that failed leaves the tab on chrome-error://; the base stays what it was
        # (or becomes the requested URL), so a path is never resolved against the error page
        wa = self._wa("chrome-error://chromewebdata/")
        wa._anchor_links_to_tab()
        assert wa._resolve_page_link("/page/2") == "https://news.example/page/2"
        wa._anchor_links_to_tab("https://shop.example/")
        assert wa._resolve_page_link("/page/2") == "https://shop.example/page/2"

    def test_a_snapshot_anchors_the_base_to_the_page_the_browser_shows(self):
        # the bug: browse_act's clicks navigated the tab, the snapshot printed the new page's
        # links, and a path read off it resolved against the page the last open_url fetched
        wa = self._wa("https://shop.example/catalog?page=1")
        wa._browse_snapshot_frame_ids, wa._browse_snapshot_labels = [], []
        with patch.object(WebAccess, "_async_sync_popup_targets", new=AsyncMock()), \
                patch.object(WebAccess, "_evaluate", new=AsyncMock(return_value=None)):
            asyncio.run(wa._async_take_snapshot())
        assert wa._resolve_page_link("/page/2") == "https://shop.example/page/2"

    def test_browse_open_resolves_a_path_against_the_page_it_opened_last(self):
        wa = self._wa("https://shop.example/catalog")
        opened = []
        with patch.object(wa, "_run_browse_op", side_effect=lambda label, coro, _err: (coro.close(), opened.append(label))[1]):
            wa._link_base_url = "https://shop.example/catalog"
            wa.browse_open("/page/2")
        assert opened == ["browse_open(https://shop.example/page/2)"]
