import re

from .MemoryTab import MemoryTab
from assistant.Procedures import Procedures


class ProceduresTab(MemoryTab):
    """Every procedure in full, each under its name."""

    TITLE = "Procedures"
    SECTIONS_LABEL = "Procedures"
    DOCS = Procedures

    def _live_text(self) -> str:
        names = self.DOCS.names()
        if not names:
            return self.DOCS.EMPTY + "\n"
        parts = []
        for name in names:
            text = (self.DOCS.read(name) or "").strip()
            # The tab's "## name" replaces the file's own "# name"; every heading in its body
            # sits below that.
            body = text.removeprefix(f"# {name}\n").removeprefix("# ")
            body = re.sub(r"(?m)^(#+)(?= )", lambda m: "#" * max(len(m.group(1)) + 1, 3), body)
            parts += [f"## {name}", "", self._body(body).strip(), ""]
        return "\n".join(parts) + "\n"

    def _body(self, body: str) -> str:
        # The curator's "## Notes" is shown lower-case, like the procedure names around it; the
        # file keeps its spelling.
        return re.sub(rf"(?m)^#{re.escape(Procedures.NOTES_HEADING)}$", f"#{Procedures.NOTES_HEADING.lower()}", body)
