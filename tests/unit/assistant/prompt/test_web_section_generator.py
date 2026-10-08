"""WebSectionGenerator: the WEB BROWSER section the `web` toolkit carries says that Chrome is
missing, and says nothing at all while it is there."""
from unittest.mock import patch

import pytest

from assistant.commands import Toolkits
from assistant.prompt.sections.WebSectionGenerator import WebSectionGenerator

MODULE = "assistant.prompt.sections.WebSectionGenerator.WebSectionGenerator"


@pytest.fixture
def section():
    return WebSectionGenerator(prompt_manager=None)


@pytest.mark.unit
class TestFetchContent:
    def test_nothing_to_say_while_chrome_is_installed(self, section):
        with patch(f"{MODULE}.chrome_binary", return_value="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"):
            assert section._fetch_content() == ""

    def test_without_chrome_the_llm_hears_what_is_missing_and_what_fixes_it(self, section):
        with patch(f"{MODULE}.chrome_binary", return_value=None):
            text = section._fetch_content()
        assert "Chrome" in text and "server" in text
        assert "browse_" in text and "open_url" in text

    def test_the_lookup_is_web_access_own(self, section):
        with patch("assistant.tools.WebAccess.WebAccess._find_chrome_binary", return_value=None) as find:
            assert section._fetch_content() == WebSectionGenerator.NO_CHROME
        find.assert_called_once()


@pytest.mark.unit
class TestCarriedByTheWebToolkit:
    def test_the_web_toolkit_carries_it_under_its_own_heading(self):
        assert Toolkits.CARRIED_SECTIONS["web"] == "web"
        rendered = Toolkits.render_sections(Toolkits.by_name("web"), {"web": WebSectionGenerator.NO_CHROME})
        assert rendered.startswith("## WEB BROWSER\n") and WebSectionGenerator.NO_CHROME in rendered

    def test_an_installed_chrome_leaves_no_heading_behind(self):
        assert Toolkits.render_sections(Toolkits.by_name("web"), {"web": ""}) == ""
        assert Toolkits.render_sections(Toolkits.by_name("web"), {}) == ""


@pytest.mark.unit
class TestPollLogging:
    """The section is polled, so an unchanged verdict must not put a line in the log per poll."""

    def test_an_unchanged_verdict_is_logged_once(self, section):
        with patch(f"{MODULE}.chrome_binary", return_value=None), \
                patch("assistant.prompt.sections.WebSectionGenerator.info") as info:
            texts = {section._fetch_content() for _ in range(3)}
        assert info.call_count == 1
        assert texts == {WebSectionGenerator.NO_CHROME}

    def test_a_chrome_installed_while_the_server_runs_is_logged_once_more(self, section):
        chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        with patch(f"{MODULE}.chrome_binary", side_effect=[None, None, chrome, chrome]), \
                patch("assistant.prompt.sections.WebSectionGenerator.info") as info:
            texts = [section._fetch_content() for _ in range(4)]
        assert info.call_count == 2
        assert texts == [WebSectionGenerator.NO_CHROME, WebSectionGenerator.NO_CHROME, "", ""]
