"""The WEB BROWSER section the `web` toolkit carries: empty while a Chrome/Chromium binary is on
the server, and the fact that none is when it is not. With it the LLM knows why pages come back
thin and the browse commands fail, and can tell the user what to install. Polled, so an install
made while the server runs reaches the LLM as a delta note."""
from ..PromptSectionGenerator import PromptSectionGenerator
from common.logging import info


class WebSectionGenerator(PromptSectionGenerator):
    """Says so when no Chrome is installed for Web Access; says nothing otherwise."""

    UPDATE_INTERVAL = 60

    NO_CHROME = ("Google Chrome is NOT installed on this server, so there is no browser: `open_url` falls back to a "
                 "plain download, which comes back empty or blocked on pages that render with JavaScript or sit "
                 "behind a bot wall, and the browse_* commands cannot run at all. Installing Google Chrome on the "
                 "server fixes it — nothing to sign in to or configure, the next web command picks it up.")

    _UNPOLLED = object()

    def __init__(self, prompt_manager):
        super().__init__(prompt_manager, "WebBrowser")
        self._binary = self._UNPOLLED    # the last poll's binary; None once a poll found none

    @staticmethod
    def chrome_binary() -> str | None:
        from assistant.tools.WebAccess import WebAccess
        return WebAccess._find_chrome_binary()

    def _fetch_content(self) -> str:
        """Polled, so the verdict is logged only when it changes from the last poll."""
        binary = self.chrome_binary()
        if binary != self._binary:
            self._binary = binary
            if binary:
                info(f"Web section: Chrome at {binary} — nothing to tell the LLM")
            else:
                info("Web section: no Chrome/Chromium binary on this machine — the web toolkit says so")
        return "" if binary else self.NO_CHROME
