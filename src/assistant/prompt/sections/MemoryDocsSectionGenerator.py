from ..PromptSectionGenerator import PromptSectionGenerator
from common.logging import info, trace


class MemoryDocsSectionGenerator(PromptSectionGenerator):
    """Renders the index of one `MemoryDocs` store — one line per file, read from the files —
    and re-renders as soon as a file is saved, changed or deleted."""

    docs = None
    title = ""

    def __init__(self, prompt_manager):
        super().__init__(prompt_manager, self.title)
        self.last_version = None

    def _start_monitoring(self):
        """Check the store's version every second."""
        while not self._stop_event.is_set():
            try:
                version = self.docs.version()
                if version != self.last_version:
                    self._update_content()
                    self.last_version = version
            except Exception as e:
                trace(f"Error checking the {self.docs.PLURAL}: {e}")
            self._wait(1)

    def _fetch_content(self) -> str:
        info(f"{self.title} section: {len(self.docs.names())} {self.docs.NOUN}(s)")
        return self.docs.render_index()
