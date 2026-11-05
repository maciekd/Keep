"""The extractor's links: tracking parameters stripped, a repeated link rendered once."""
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
        assert '<a href="https://www.booking.com/h?id=5">Hotel</a>' in text
        assert "aid=" not in text

    def test_a_repeated_link_is_rendered_once_and_its_later_texts_stay(self):
        text = WebAccess.extract_text_with_links(
            _page('<a href="/h?aid=1&id=5">Hotel</a> <a href="/h?aid=2&id=5">Hotel</a> '
                  '<a href="/h?id=5">SEK 35,622</a> <a href="/other">Other</a>'),
            "https://www.booking.com/")
        assert text.count("<a href=") == 2
        assert text.count('<a href="https://www.booking.com/h?id=5">') == 1
        assert "SEK 35,622" in text
        assert '<a href="https://www.booking.com/other">Other</a>' in text

    def test_distinct_links_are_all_kept(self):
        text = WebAccess.extract_text_with_links(
            _page('<a href="/a">A</a> <a href="/b">B</a> <a href="/a#x">A again</a>'),
            "https://example.com/")
        assert text.count("<a href=") == 3

    def test_the_seen_set_is_per_page(self):
        page = _page('<a href="/a">A</a>')
        first = WebAccess.extract_text_with_links(page, "https://example.com/")
        second = WebAccess.extract_text_with_links(page, "https://example.com/")
        assert first == second and "<a href=" in second
