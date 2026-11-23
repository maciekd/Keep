from .ProceduresTab import ProceduresTab
from assistant.Memos import Memos


class MemosTab(ProceduresTab):
    """Every memo in full, each under its name."""

    TITLE = "Memos"
    SECTIONS_LABEL = "Memos"
    DOCS = Memos

    def _body(self, body: str) -> str:
        return body
