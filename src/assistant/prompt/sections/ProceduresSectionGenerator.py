from .MemoryDocsSectionGenerator import MemoryDocsSectionGenerator
from assistant.Procedures import Procedures


class ProceduresSectionGenerator(MemoryDocsSectionGenerator):
    """Renders the index of the saved procedures and re-renders as soon as one is saved, annotated or deleted."""

    docs = Procedures
    title = "Procedures"
