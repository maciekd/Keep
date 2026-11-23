from .MemoryDocsSectionGenerator import MemoryDocsSectionGenerator
from assistant.Memos import Memos


class MemosSectionGenerator(MemoryDocsSectionGenerator):
    """Renders the index of the saved memos and re-renders as soon as one is saved, replaced or deleted."""

    docs = Memos
    title = "Memos"
